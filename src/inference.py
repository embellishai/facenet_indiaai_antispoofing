"""
Inference script for face recognition using finetuned buffalo_l model
Handles face detection, alignment, and recognition
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    setup_logging,
    load_config,
    get_device,
    load_checkpoint
)
from src.model import FaceRecognitionModel


logger = logging.getLogger('insightface_finetune')


class FaceRecognizer:
    """
    Face recognition system with detection, alignment, and recognition
    """
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        threshold: float = 0.6,
        image_size: Tuple[int, int] = (112, 112)
    ):
        """
        Initialize face recognizer
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            threshold: Similarity threshold for recognition
            image_size: Input image size for model
        """
        self.device = device
        self.threshold = threshold
        self.image_size = image_size
        
        logger.info('🚀 ***** Starting face recognizer initialization...')
        
        # Load model checkpoint
        self.checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model information
        self.num_classes = self.checkpoint['num_classes']
        self.embedding_size = self.checkpoint.get('embedding_size', 512)
        self.identity_to_label = self.checkpoint['identity_to_label']
        self.label_to_identity = self.checkpoint['label_to_identity']
        
        # Convert label_to_identity keys to int (they may be saved as strings)
        self.label_to_identity = {int(k): v for k, v in self.label_to_identity.items()}
        
        logger.info(f'📊 {self.num_classes} identities loaded')
        
        # Create model
        self.model = FaceRecognitionModel(
            num_classes=self.num_classes,
            embedding_size=self.embedding_size,
            depth=100,
            dropout=0.0
        )
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        logger.info('✅ Model loaded successfully')
        
        # Create preprocessing transform
        self.transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
        # Compute embeddings for all known identities (for gallery matching)
        self.gallery_embeddings = None
        self.gallery_labels = None
        
        logger.info('✅ ***** Face recognizer initialization done.')
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from image
        
        Args:
            image: Face image (BGR format)
            
        Returns:
            Normalized face embedding
        """
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            embedding, _ = self.model(image_tensor, labels=None)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        embedding_np = embedding.cpu().numpy()[0]
        
        return embedding_np
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def recognize_face(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Recognize identity from face image
        
        Args:
            image: Face image (BGR format)
            
        Returns:
            Recognition result dictionary
        """
        # Extract embedding
        embedding = self.extract_embedding(image)
        
        # Compare with all known identities using model's classification head
        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            weight_normalized = F.normalize(self.model.loss_head.weight, p=2, dim=1)
            similarities = torch.matmul(embedding_tensor, weight_normalized.t())
            
            # Get top predictions
            top_k = min(5, self.num_classes)
            top_similarities, top_indices = torch.topk(similarities, k=top_k, dim=1)
            
            top_similarities = top_similarities.cpu().numpy()[0]
            top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        results = []
        for i in range(top_k):
            label = int(top_indices[i])
            similarity = float(top_similarities[i])
            identity = self.label_to_identity.get(label, 'Unknown')
            
            results.append({
                'identity': identity,
                'label': label,
                'similarity': similarity,
                'confidence': similarity * 100.0
            })
        
        # Best match
        best_match = results[0]
        best_match['is_recognized'] = best_match['similarity'] >= self.threshold
        
        return {
            'best_match': best_match,
            'top_matches': results,
            'embedding': embedding.tolist()
        }
    
    def recognize_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Recognize face from image file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Recognition result dictionary
        """
        logger.info(f'🔍 Processing image: {image_path}')
        
        # Load image
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f'❌ Failed to load image: {image_path}')
            return {'error': 'Failed to load image'}
        
        # Recognize face
        result = self.recognize_face(image)
        result['image_path'] = image_path
        
        return result
    
    def recognize_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Recognize faces from multiple images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of recognition results
        """
        logger.info(f'🔍 ***** Starting batch recognition for {len(image_paths)} images...')
        
        results = []
        
        for image_path in image_paths:
            result = self.recognize_from_path(image_path)
            results.append(result)
        
        logger.info('✅ ***** Batch recognition done.')
        
        return results
    
    def verify_faces(self, image_path1: str, image_path2: str) -> Dict[str, Any]:
        """
        Verify if two face images belong to the same person
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            
        Returns:
            Verification result dictionary
        """
        logger.info(f'🔍 Verifying: {image_path1} vs {image_path2}')
        
        # Load images
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        
        if image1 is None or image2 is None:
            logger.error('❌ Failed to load one or both images')
            return {'error': 'Failed to load images'}
        
        # Extract embeddings
        embedding1 = self.extract_embedding(image1)
        embedding2 = self.extract_embedding(image2)
        
        # Compute similarity
        similarity = self.compute_similarity(embedding1, embedding2)
        
        is_same_person = similarity >= self.threshold
        
        result = {
            'similarity': float(similarity),
            'confidence': float(similarity * 100.0),
            'is_same_person': is_same_person,
            'threshold': self.threshold,
            'image1': image_path1,
            'image2': image_path2
        }
        
        logger.info(f'🎯 Similarity: {similarity:.4f} ({result["confidence"]:.2f}%) - {"✅ Same" if is_same_person else "❌ Different"}')
        
        return result


def visualize_recognition_result(
    image_path: str,
    result: Dict[str, Any],
    output_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Visualize recognition result on image
    
    Args:
        image_path: Path to input image
        result: Recognition result dictionary
        output_path: Path to save annotated image (optional)
        show: Show image in window
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        logger.error(f'❌ Failed to load image: {image_path}')
        return
    
    # Get best match
    best_match = result.get('best_match', {})
    identity = best_match.get('identity', 'Unknown')
    confidence = best_match.get('confidence', 0.0)
    is_recognized = best_match.get('is_recognized', False)
    
    # Add text to image
    text = f'{identity}: {confidence:.1f}%'
    color = (0, 255, 0) if is_recognized else (0, 0, 255)
    
    # Add background rectangle for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Draw rectangle
    cv2.rectangle(
        image,
        (10, 10),
        (20 + text_size[0], 40 + text_size[1]),
        color,
        -1
    )
    
    # Draw text
    cv2.putText(
        image,
        text,
        (15, 35 + text_size[1]),
        font,
        font_scale,
        (255, 255, 255),
        thickness
    )
    
    # Save or show
    if output_path:
        cv2.imwrite(output_path, image)
        logger.info(f'💾 Annotated image saved to {output_path}')
    
    if show:
        cv2.imshow('Recognition Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point for inference"""
    parser = argparse.ArgumentParser(description='Face recognition inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--images', type=str, nargs='+', help='Paths to multiple input images')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--verify', type=str, nargs=2, help='Verify two face images')
    parser.add_argument('--threshold', type=float, default=0.6, help='Recognition threshold')
    parser.add_argument('--output', type=str, help='Output file for results (JSON)')
    parser.add_argument('--visualize', action='store_true', help='Visualize results on images')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Set up logging
    logger_instance = setup_logging(
        log_dir='logs',
        log_level='INFO',
        console=True,
        file=False
    )
    
    # Get device
    device = get_device(device_name=args.device)
    
    # Create recognizer
    recognizer = FaceRecognizer(
        model_path=args.model,
        device=device,
        threshold=args.threshold
    )
    
    # Process based on input type
    results = None
    
    if args.verify:
        # Face verification mode
        result = recognizer.verify_faces(args.verify[0], args.verify[1])
        results = [result]
        
        # Print result
        print(f'\n{"=" * 80}')
        print('FACE VERIFICATION RESULT')
        print(f'{"=" * 80}')
        print(f'Image 1: {result["image1"]}')
        print(f'Image 2: {result["image2"]}')
        print(f'Similarity: {result["similarity"]:.4f} ({result["confidence"]:.2f}%)')
        print(f'Same Person: {"✅ YES" if result["is_same_person"] else "❌ NO"}')
        print(f'{"=" * 80}\n')
        
    elif args.image:
        # Single image recognition
        result = recognizer.recognize_from_path(args.image)
        results = [result]
        
        # Print result
        print(f'\n{"=" * 80}')
        print('FACE RECOGNITION RESULT')
        print(f'{"=" * 80}')
        print(f'Image: {result["image_path"]}')
        
        best_match = result['best_match']
        print(f'\nBest Match:')
        print(f'  Identity: {best_match["identity"]}')
        print(f'  Confidence: {best_match["confidence"]:.2f}%')
        print(f'  Recognized: {"✅ YES" if best_match["is_recognized"] else "❌ NO"}')
        
        print(f'\nTop 5 Matches:')
        for i, match in enumerate(result['top_matches'], 1):
            print(f'  {i}. {match["identity"]}: {match["confidence"]:.2f}%')
        
        print(f'{"=" * 80}\n')
        
        # Visualize if requested
        if args.visualize:
            output_path = args.image.replace('.', '_annotated.') if not args.output else args.output
            visualize_recognition_result(args.image, result, output_path)
        
    elif args.images:
        # Multiple images recognition
        results = recognizer.recognize_batch(args.images)
        
        # Print results
        print(f'\n{"=" * 80}')
        print(f'BATCH RECOGNITION RESULTS ({len(results)} images)')
        print(f'{"=" * 80}\n')
        
        for i, result in enumerate(results, 1):
            best_match = result['best_match']
            print(f'{i}. {Path(result["image_path"]).name}')
            print(f'   → {best_match["identity"]}: {best_match["confidence"]:.2f}%')
        
        print(f'\n{"=" * 80}\n')
        
    elif args.image_dir:
        # Directory of images
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = [
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        results = recognizer.recognize_batch(image_paths)
        
        # Print summary
        print(f'\n{"=" * 80}')
        print(f'DIRECTORY RECOGNITION RESULTS ({len(results)} images)')
        print(f'{"=" * 80}\n')
        
        for i, result in enumerate(results, 1):
            best_match = result['best_match']
            print(f'{i}. {Path(result["image_path"]).name}')
            print(f'   → {best_match["identity"]}: {best_match["confidence"]:.2f}%')
        
        print(f'\n{"=" * 80}\n')
    
    else:
        parser.print_help()
        return
    
    # Save results to JSON if output specified
    if args.output and results:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'💾 Results saved to {args.output}')


if __name__ == '__main__':
    main()

