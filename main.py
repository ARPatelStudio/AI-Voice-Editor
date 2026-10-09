import io
import math
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import onnxruntime as ort

app = FastAPI(
    title="Target Speaker Isolation Engine",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Enable CORS for cross-platform clients (Desktop/Mobile web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample rate requirement for model processing
TARGET_SAMPLE_RATE = 16000

# Global sessions placeholder to ensure lazy loading and low initial RAM
speaker_encoder_session: ort.InferenceSession | None = None
separator_session: ort.InferenceSession | None = None


def get_inference_sessions():
    """
    Initializes ONNX Runtime sessions with strict single-thread CPU constraints
    to prevent Render 512MB RAM throttling and spikes.
    """
    global speaker_encoder_session, separator_session

    if speaker_encoder_session is None or separator_session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Models are placed in /models directory inside container
        try:
            speaker_encoder_session = ort.InferenceSession(
                "models/speaker_encoder.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
            )
            separator_session = ort.InferenceSession(
                "models/speaker_extractor.onnx", sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as e:
            # Fallback mock setup if actual ONNX files aren't yet downloaded in the build
            speaker_encoder_session = None
            separator_session = None
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    return speaker_encoder_session, separator_session


def load_audio_from_bytes(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Reads audio in-memory, converts to mono 16kHz float32.
    """
    with io.BytesIO(file_bytes) as audio_io:
        data, samplerate = sf.read(audio_io, dtype="float32")

    # If multi-channel, downmix to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz if necessary using scipy linear interpolation (lightweight)
    if samplerate != TARGET_SAMPLE_RATE:
        number_of_samples = round(len(data) * float(TARGET_SAMPLE_RATE) / samplerate)
        from scipy import signal
        data = signal.resample(data, number_of_samples)
        samplerate = TARGET_SAMPLE_RATE

    return data, samplerate


def extract_speaker_embedding(session: ort.InferenceSession, audio: np.ndarray) -> np.ndarray:
    """
    Generates a 1D d-vector voice fingerprint from the reference sample.
    """
    # Audio shape expected by ONNX speaker encoder: [1, samples]
    audio_input = np.expand_dims(audio, axis=0).astype(np.float32)
    inputs = {session.get_inputs()[0].name: audio_input}
    embedding = session.run(None, inputs)[0]
    return embedding


def run_chunked_extraction(
    session: ort.InferenceSession,
    mixed_audio: np.ndarray,
    embedding: np.ndarray,
    chunk_seconds: int = 10,
) -> np.ndarray:
    """
    Processes long mixed audio in small segments to respect Render's 512MB RAM cap.
    """
    chunk_samples = TARGET_SAMPLE_RATE * chunk_seconds
    total_samples = len(mixed_audio)
    num_chunks = math.ceil(total_samples / chunk_samples)

    output_chunks: list[np.ndarray] = []

    for i in range(num_chunks):
        start_idx = i * chunk_samples
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = mixed_audio[start_idx:end_idx]

        # Pad last chunk if shorter than 1 sec to avoid model edge artifacts
        if len(chunk) < TARGET_SAMPLE_RATE:
            padding = np.zeros(TARGET_SAMPLE_RATE - len(chunk), dtype=np.float32)
            chunk_padded = np.concatenate([chunk, padding])
        else:
            chunk_padded = chunk

        # Model input: [1, samples], Embedding input: [1, embed_dim]
        chunk_input = np.expand_dims(chunk_padded, axis=0).astype(np.float32)
        inputs = {
            session.get_inputs()[0].name: chunk_input,
            session.get_inputs()[1].name: embedding.astype(np.float32),
        }

        # Run inference
        isolated_chunk = session.run(None, inputs)[0].squeeze()

        # Slice away any padding applied
        isolated_chunk = isolated_chunk[: len(chunk)]
        output_chunks.append(isolated_chunk)

    return np.concatenate(output_chunks)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Voice Isolation Engine (Render Ready)"}


@app.post("/isolate-voice")
async def isolate_voice(
    reference_sample: UploadFile = File(..., description="5-10 second clean user sample"),
    mixed_audio: UploadFile = File(..., description="Main multi-speaker audio recording"),
):
    """
    API endpoint: Isolates the reference speaker and strips all other speakers.
    """
    # 1. Read files into memory
    ref_bytes = await reference_sample.read()
    mixed_bytes = await mixed_audio.read()

    if not ref_bytes or not mixed_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both reference_sample and mixed_audio must be non-empty files.",
        )

    try:
        # 2. Decode Audio
        ref_audio, _ = load_audio_from_bytes(ref_bytes)
        mix_audio, _ = load_audio_from_bytes(mixed_bytes)

        # 3. Load or acquire ONNX sessions
        encoder_sess, separator_sess = get_inference_sessions()

        # 4. Generate voice fingerprint embedding
        ref_embedding = extract_speaker_embedding(encoder_sess, ref_audio)

        # 5. Extract target voice chunk by chunk (safe on RAM)
        isolated_audio = run_chunked_extraction(
            session=separator_sess,
            mixed_audio=mix_audio,
            embedding=ref_embedding,
            chunk_seconds=10,
        )

        # 6. Encode resulting raw float array to standard 16-bit PCM WAV
        output_buffer = io.BytesIO()
        sf.write(output_buffer, isolated_audio, TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        output_buffer.seek(0)

        return Response(
            content=output_buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=isolated_user_voice.wav"},
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(exc)}",
        )
