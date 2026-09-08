"""
identity_manager.py

Persistent Person Identity and Re-Identification (Re-ID) Layer.
Maintains persistent person identities (P001, P002, ...) across disappearances
and re-entries using genuine appearance embedding similarity.

Key Architecture:
    - Deep neural appearance feature extractor (MobileNetV3 backbone + multi-region spatial anatomical color features)
    - In-memory registry of PersistentPerson entities
    - Cosine similarity appearance matching
    - Anti-false-match safeguards (reid_similarity_threshold, ambiguity margin, EMA temporal smoothing)
    - PersonID string wrapper ensuring zero-crash formatting compatibility with existing UI
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


class PersonID(str):
    """
    String representation of a persistent person ID (e.g. 'P001') that supports
    formatting specifications like {:02d} without error.

    This ensures seamless compatibility with app.py (line 504: f"{t['track_id']:02d}")
    without requiring any modifications to app.py.
    """
    def __format__(self, format_spec: str) -> str:
        return str(self)


@dataclass
class PersistentPerson:
    """
    Data structure representing a known persistent identity in the registry.
    """
    person_id: str                      # e.g., 'P001'
    appearance_embedding: np.ndarray    # L2-normalized 1D feature vector
    last_seen_time: float               # Unix timestamp
    last_seen_frame: int                # Frame index when last observed
    last_track_id: int                  # Most recent ByteTrack temporary track ID
    currently_visible: bool = True      # VISIBLE / ACTIVE vs TEMPORARILY LOST / LEFT
    sighting_count: int = 1             # Number of frames this person was visible

    def update_appearance(self, new_embedding: np.ndarray, alpha: float = 0.85):
        """
        Updates the stored appearance embedding using Exponential Moving Average (EMA)
        and re-normalizes to maintain a unit vector.
        """
        if new_embedding is None or len(new_embedding) == 0:
            return
        updated = alpha * self.appearance_embedding + (1.0 - alpha) * new_embedding
        norm = np.linalg.norm(updated)
        if norm > 1e-6:
            self.appearance_embedding = updated / norm
        self.sighting_count += 1


class AppearanceExtractor:
    """
    Lightweight appearance feature extractor.
    Combines:
      1. Pretrained PyTorch MobileNetV3 convolutional features (deep texture, silhouette, structural patterns)
      2. Multi-zone spatial anatomical color distribution (Head/Torso/Legs in HSV space with center weighting)
    Produces an L2-normalized unit feature vector for robust cosine similarity matching.
    """
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.has_neural_model = False
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            base_model = models.mobilenet_v3_small(weights=weights)
            self.backbone = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            ).to(self.device)
            self.backbone.eval()
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.has_neural_model = True
        except Exception:
            self.has_neural_model = False

    def _extract_spatial_color_descriptor(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts multi-zone anatomical color features (Head: 0-30%, Torso: 30-70%, Legs: 70-100%)
        using 2D Hue-Saturation histograms with center-weighted masks.
        """
        h, w = crop_bgr.shape[:2]
        if h < 6 or w < 6:
            return np.zeros(144, dtype=np.float32)

        # 3 vertical anatomical zones
        zones = [
            crop_bgr[0:max(1, int(h * 0.30)), :],
            crop_bgr[int(h * 0.30):max(int(h * 0.30) + 1, int(h * 0.70)), :],
            crop_bgr[int(h * 0.70):, :]
        ]

        zone_feats = []
        for zone in zones:
            zh, zw = zone.shape[:2]
            if zh == 0 or zw == 0:
                zone_feats.append(np.zeros(48, dtype=np.float32))
                continue

            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            # Center weighting to suppress background clutter near bounding box margins
            mask = np.zeros((zh, zw), dtype=np.uint8)
            pad_y, pad_x = max(1, int(zh * 0.1)), max(1, int(zw * 0.1))
            mask[pad_y:max(pad_y + 1, zh - pad_y), pad_x:max(pad_x + 1, zw - pad_x)] = 255

            # 2D HS histogram (12 Hue bins, 4 Saturation bins = 48 bins)
            hist = cv2.calcHist([hsv], [0, 1], mask, [12, 4], [0, 180, 0, 256]).flatten()
            norm = np.linalg.norm(hist)
            if norm > 1e-6:
                hist = hist / norm
            zone_feats.append(hist)

        combined_color = np.concatenate(zone_feats).astype(np.float32)
        norm = np.linalg.norm(combined_color)
        if norm > 1e-6:
            combined_color /= norm
        return combined_color

    def extract(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a fused L2-normalized appearance embedding from a person crop.
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            return None

        # 1. Spatial anatomical color descriptor (144-dim)
        color_feat = self._extract_spatial_color_descriptor(crop_bgr)

        # 2. Deep neural appearance features (576-dim)
        if self.has_neural_model:
            try:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    neural_feat = self.backbone(tensor).squeeze(0).cpu().numpy().astype(np.float32)
                neural_norm = np.linalg.norm(neural_feat)
                if neural_norm > 1e-6:
                    neural_feat /= neural_norm
            except Exception:
                neural_feat = None
        else:
            neural_feat = None

        # 3. Feature fusion & L2-normalization
        # Fuses multi-zone anatomical clothing appearance (dominant for Re-ID)
        # with deep convolutional structural patterns.
        if neural_feat is not None:
            fused = np.concatenate([color_feat * 0.85, neural_feat * 0.15])
        else:
            fused = color_feat

        norm = np.linalg.norm(fused)
        if norm > 1e-6:
            fused /= norm
            return fused
        return None


class IdentityManager:
    """
    Manages persistent person identities (P001, P002, ...) across video frames.
    Sits directly above ByteTrack to map temporary track IDs to persistent person IDs.

    Maintains:
      - active_track_to_person: dict mapping temporary ByteTrack track_id -> persistent person_id
      - registry: dict mapping person_id -> PersistentPerson
    """
    def __init__(
        self,
        reid_similarity_threshold: float = 0.70,
        reid_similarity_margin: float = 0.05,
        feature_update_alpha: float = 0.85,
        identity_lost_timeout: float = 30.0,
        device: Optional[str] = None
    ):
        self.reid_similarity_threshold = reid_similarity_threshold
        self.reid_similarity_margin = reid_similarity_margin
        self.feature_update_alpha = feature_update_alpha
        self.identity_lost_timeout = identity_lost_timeout

        self.extractor = AppearanceExtractor(device=device)
        self.registry: Dict[str, PersistentPerson] = {}
        self.active_track_to_person: Dict[int, str] = {}
        self.next_id_index: int = 1
        self.current_frame_idx: int = 0

    def _generate_next_id(self) -> str:
        """Generates the next sequential persistent person ID (P001, P002, ...)."""
        pid = f"P{self.next_id_index:03d}"
        self.next_id_index += 1
        return pid

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine similarity between two L2-normalized unit vectors."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))

    def update(self, tracked_persons: List[dict], frame: np.ndarray) -> List[dict]:
        """
        Updates the identity layer with current ByteTrack tracks and raw frame.
        Maps temporary ByteTrack Track IDs to persistent Person IDs (P001, P002, ...).

        Args:
            tracked_persons: List of dicts [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'conf': float, ...}]
            frame: Input video frame (numpy array)

        Returns:
            tracked_persons: Updated list where 'person_id' is set and 'track_id' is wrapped as PersonID.
        """
        self.current_frame_idx += 1
        current_time = time.time()

        if not tracked_persons:
            # Mark all previously active persons as temporarily lost
            for person in self.registry.values():
                person.currently_visible = False
            self.active_track_to_person.clear()
            return []

        frame_h, frame_w = frame.shape[:2]
        current_active_track_ids = set()
        assigned_p_ids_in_frame = set()

        # Step 1: Separate continuously visible tracks from new/unmapped tracks
        unmapped_tracks = []
        for track in tracked_persons:
            t_id = int(track['track_id'])
            current_active_track_ids.add(t_id)

            # Crop person image for appearance extraction
            bbox = track['bbox']
            x1 = max(0, min(frame_w - 1, int(bbox[0])))
            y1 = max(0, min(frame_h - 1, int(bbox[1])))
            x2 = max(0, min(frame_w, int(bbox[2])))
            y2 = max(0, min(frame_h, int(bbox[3])))
            crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

            embedding = self.extractor.extract(crop) if crop is not None else None
            track['_crop_embedding'] = embedding

            # Rule 1: Continuously visible person
            if t_id in self.active_track_to_person:
                p_id = self.active_track_to_person[t_id]
                assigned_p_ids_in_frame.add(p_id)
                person = self.registry[p_id]
                person.last_seen_time = current_time
                person.last_seen_frame = self.current_frame_idx
                person.last_track_id = t_id
                person.currently_visible = True

                # EMA update while continuously visible
                if embedding is not None:
                    person.update_appearance(embedding, alpha=self.feature_update_alpha)
            else:
                unmapped_tracks.append(track)

        # Step 2: Resolve identities for new / unmapped tracks
        for track in unmapped_tracks:
            t_id = int(track['track_id'])
            embedding = track.get('_crop_embedding')

            # Find candidate identities from registry that are NOT currently visible in this frame
            candidates = [
                p for p in self.registry.values()
                if (not p.currently_visible) and (p.person_id not in assigned_p_ids_in_frame)
            ]

            # Filter candidates within lost timeout
            eligible_candidates = [
                p for p in candidates
                if (current_time - p.last_seen_time) <= self.identity_lost_timeout
            ]

            assigned_p_id: Optional[str] = None

            # Calculate appearance similarity against all eligible candidates
            if embedding is not None and eligible_candidates:
                scored_candidates: List[Tuple[float, PersistentPerson]] = []
                for candidate in eligible_candidates:
                    sim = self.compute_similarity(embedding, candidate.appearance_embedding)
                    scored_candidates.append((sim, candidate))

                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                best_sim, best_candidate = scored_candidates[0]
                second_best_sim = scored_candidates[1][0] if len(scored_candidates) > 1 else -1.0

                # Re-entry matching with threshold & ambiguity safeguard
                if best_sim >= self.reid_similarity_threshold:
                    # Ambiguity protection: ensure top match is clear of the runner-up
                    if (second_best_sim < 0) or ((best_sim - second_best_sim) >= self.reid_similarity_margin):
                        # Rule 3: Re-entry recognized! Restore persistent Person ID
                        assigned_p_id = best_candidate.person_id
                        best_candidate.last_seen_time = current_time
                        best_candidate.last_seen_frame = self.current_frame_idx
                        best_candidate.last_track_id = t_id
                        best_candidate.currently_visible = True
                        best_candidate.update_appearance(embedding, alpha=self.feature_update_alpha)

            # Rule 4: New person (no sufficiently strong match or ambiguous)
            if assigned_p_id is None:
                assigned_p_id = self._generate_next_id()
                # If no embedding could be extracted, initialize with random unit vector
                init_emb = embedding if embedding is not None else np.random.randn(144).astype(np.float32)
                init_norm = np.linalg.norm(init_emb)
                if init_norm > 1e-6:
                    init_emb = init_emb / init_norm

                new_person = PersistentPerson(
                    person_id=assigned_p_id,
                    appearance_embedding=init_emb,
                    last_seen_time=current_time,
                    last_seen_frame=self.current_frame_idx,
                    last_track_id=t_id,
                    currently_visible=True
                )
                self.registry[assigned_p_id] = new_person

            self.active_track_to_person[t_id] = assigned_p_id
            assigned_p_ids_in_frame.add(assigned_p_id)

        # Step 3: Update visibility states for persistent persons not in current frame
        for person in self.registry.values():
            if person.person_id not in assigned_p_ids_in_frame:
                person.currently_visible = False

        # Clean up active_track_to_person to only retain current frame's tracks
        self.active_track_to_person = {
            tid: pid for tid, pid in self.active_track_to_person.items()
            if tid in current_active_track_ids
        }

        # Step 4: Inject persistent identity into tracked_persons list
        for track in tracked_persons:
            raw_t_id = int(track['track_id'])
            p_id_str = self.active_track_to_person.get(raw_t_id, f"P{raw_t_id:03d}")

            track['person_id'] = p_id_str
            track['temporary_track_id'] = raw_t_id
            # PersonID wrapper guarantees f"{t['track_id']:02d}" prints "P001" without crashing
            track['track_id'] = PersonID(p_id_str)
            track.pop('_crop_embedding', None)

        return tracked_persons

    def get_total_unique_count(self) -> int:
        """Returns the total number of unique persistent person identities observed."""
        return len(self.registry)

    def reset(self):
        """Resets the identity registry and active mappings."""
        self.registry.clear()
        self.active_track_to_person.clear()
        self.next_id_index = 1
        self.current_frame_idx = 0
