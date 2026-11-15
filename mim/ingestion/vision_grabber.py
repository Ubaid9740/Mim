from pathlib import Path
from typing import Optional, Tuple
import mss
import numpy as np
from PIL import Image
from mim.config import CONFIG


class VisionGrabber:
    """
    Captures desktop frames via MSS/DXGI, computes 64-bit perceptual difference hashes (dHash)
    in RAM, drops static frames, and writes changed frames as compressed WebP files.
    """

    def __init__(
        self,
        frames_dir: Path = CONFIG.frames_dir,
        dhash_threshold: int = CONFIG.dhash_threshold,
        quality: int = CONFIG.jpeg_quality,
    ):
        self.frames_dir = frames_dir
        self.dhash_threshold = dhash_threshold
        self.quality = quality
        self._last_hash_int: Optional[int] = None
        self._sct: Optional[mss.mss] = None

    def _get_sct(self):
        """Lazily initializes the thread-local MSS screen capture context."""
        if self._sct is None:
            factory = getattr(mss, "MSS", mss.mss)
            self._sct = factory()
        return self._sct

    @staticmethod
    def compute_dhash(img: Image.Image) -> Tuple[int, str]:
        """
        Computes a 64-bit horizontal gradient difference hash (dHash).
        Resizes the image to 9x8 grayscale and compares adjacent horizontal pixels.
        Returns: (64-bit integer hash, 16-char hex string)
        """
        small = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        pixels = np.asarray(small, dtype=np.int16)
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        hash_int = 0
        for bit in diff.flatten():
            hash_int = (hash_int << 1) | int(bit)
        return hash_int, f"{hash_int:016x}"

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Computes the bitwise Hamming distance between two 64-bit integer hashes."""
        return (hash1 ^ hash2).bit_count()

    def capture_delta_frame(
        self,
        frame_id: str,
        force: bool = False
    ) -> Optional[Tuple[str, str, int]]:
        """
        Captures the primary monitor and checks if the visual delta exceeds `dhash_threshold`.
        - If static (and not forced), drops the buffer in RAM and returns None.
        - If changed, compresses to WebP on disk and returns (relative_image_path, dhash_hex, hamming_dist).
        """
        sct = self._get_sct()
        # Monitor 1 is the primary display in MSS (Monitor 0 is the virtual combined desktop)
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
        raw_shot = sct.grab(monitor)

        # Convert BGRA raw buffer to RGB PIL Image in memory
        img = Image.frombytes("RGB", raw_shot.size, raw_shot.bgra, "raw", "BGRX")
        current_hash_int, current_hash_hex = self.compute_dhash(img)

        distance = 64
        if self._last_hash_int is not None:
            distance = self.hamming_distance(self._last_hash_int, current_hash_int)
            if not force and distance < self.dhash_threshold:
                # Static screen detected: discard immediately in RAM
                return None

        # Commit changed frame to disk as high-efficiency WebP
        self._last_hash_int = current_hash_int
        filename = f"{frame_id}.webp"
        out_path = self.frames_dir / filename
        img.save(out_path, format="WEBP", quality=self.quality, method=4)

        rel_path = f"data/frames/{filename}"
        return rel_path, current_hash_hex, distance

    def reset_baseline(self) -> None:
        """Resets the delta hash baseline (used after unpausing the Eye Sentinel)."""
        self._last_hash_int = None

    def close(self) -> None:
        """Releases the underlying OS screen capture handles."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None