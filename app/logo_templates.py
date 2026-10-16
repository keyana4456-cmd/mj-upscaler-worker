"""Recognize reference watermarks and transfer their approved shape masks.

Features establish identity and geometry; no bounding-box expansion is used.
Unknown logos remain the responsibility of the normal detector/outline tool.
"""
from pathlib import Path
import json
import cv2
import numpy as np


class LogoTemplates:
    def __init__(self, directory=None):
        directory = Path(directory or Path(__file__).parent / 'watermark_templates')
        self.sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.02)
        self.matcher = cv2.BFMatcher()
        self.references = []
        for config in sorted(directory.glob('*.json')):
            data = json.loads(config.read_text(encoding='utf-8'))
            image = cv2.imread(str(directory / data['image']), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(directory / data['mask']), cv2.IMREAD_GRAYSCALE)
            feature_mask = cv2.imread(str(directory / data['features']), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or feature_mask is None or image.shape != mask.shape or image.shape != feature_mask.shape:
                raise ValueError(f'Invalid watermark reference: {config.name}')
            points, descriptors = self.sift.detectAndCompute(image, feature_mask)
            if descriptors is not None and len(points) >= 12:
                graphic = cv2.imread(str(directory / data['graphic']), cv2.IMREAD_GRAYSCALE) if data.get('graphic') else None
                if graphic is None or graphic.shape != image.shape:
                    raise ValueError(f'Missing graphic verification mask: {config.name}')
                dark = (graphic > 0) & (image < 35)
                bright = (graphic > 0) & (image > 140)
                ring = (cv2.dilate(graphic, np.ones((15,15), np.uint8)) > 0) & (mask == 0)
                self.references.append((data['name'], image.shape, mask, points, descriptors, dark, bright, ring))

    def detect(self, image):
        mask = np.zeros(image.shape[:2], np.uint8)
        if not self.references:
            return mask, []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        reduction = min(1.0, 1600/max(gray.shape))
        if reduction < 1:
            gray = cv2.resize(gray, None, fx=reduction, fy=reduction, interpolation=cv2.INTER_AREA)
        points, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return mask, []
        found = []
        for name, shape, reference_mask, reference_points, reference_descriptors, dark, bright, ring in self.references:
            matches = self.matcher.knnMatch(reference_descriptors, descriptors, k=2)
            good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .68 * pair[1].distance]
            if len(good) < 12:
                continue
            source = np.float32([reference_points[m.queryIdx].pt for m in good])
            target = np.float32([points[m.trainIdx].pt for m in good]) / reduction
            transform, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                ransacReprojThreshold=2.5, maxIters=2000, confidence=.995)
            if transform is None or inliers is None:
                continue
            selected = inliers.ravel().astype(bool)
            if selected.sum() < 12 or selected.mean() < .6:
                continue
            support = source[selected]
            height, width = shape
            # Distributed matches reject a few repeated letters or desk knobs.
            if np.ptp(support[:, 0]) < width*.45 or np.ptp(support[:, 1]) < height*.1:
                continue
            scale = float(np.hypot(transform[0, 0], transform[1, 0]))
            angle = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
            if not .2 <= scale <= 4 or abs(angle) > 5:
                continue
            # Matching lettering alone is not evidence that its adjacent graphic
            # is present. Verify this reference's dark silhouette against its
            # surroundings before transferring the user-approved graphic shape.
            aligned = cv2.warpAffine(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), transform,
                (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            if not dark.any() or not bright.any() or not ring.any():
                continue
            if float(aligned[bright].mean()) - float(aligned[dark].mean()) < 40:
                continue
            if float(aligned[ring].mean()) - float(aligned[dark].mean()) < 12:
                continue
            warped = cv2.warpAffine(reference_mask, transform, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            # Reject partial off-frame matches and implausibly large repairs.
            expected = np.count_nonzero(reference_mask)*scale*scale
            area = np.count_nonzero(warped)
            if area < expected*.85 or area > image.shape[0]*image.shape[1]*.12:
                continue
            mask |= warped
            found.append(dict(name=name, inliers=int(selected.sum()), scale=scale,
                              rotation_degrees=angle, area_pixels=int(area)))
        return mask, found
