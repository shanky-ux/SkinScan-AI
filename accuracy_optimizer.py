import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class AccuracyOptimizer:
    """
    Advanced accuracy optimization techniques for skin disease detection
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def confidence_calibration(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Apply temperature scaling for better confidence calibration
        """
        return F.softmax(logits / temperature, dim=1)
    
    def ensemble_prediction(self, images: List[torch.Tensor]) -> Tuple[int, float, np.ndarray]:
        """
        Ensemble prediction from multiple image augmentations
        """
        predictions = []
        
        with torch.no_grad():
            self.model.eval()
            
            for img in images:
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                    
                output = self.model(img.to(self.device))
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())
        
        # Average ensemble predictions
        ensemble_prob = np.mean(predictions, axis=0)[0]
        
        # Get final prediction
        predicted_idx = np.argmax(ensemble_prob)
        confidence = ensemble_prob[predicted_idx] * 100
        
        return predicted_idx, confidence, ensemble_prob
    
    def monte_carlo_dropout(self, image: torch.Tensor, n_samples: int = 10) -> Tuple[int, float, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(image.unsqueeze(0).to(self.device))
                prob = F.softmax(output, dim=1)
                predictions.append(prob.cpu().numpy())
        
        self.model.eval()  # Back to eval mode
        
        # Calculate mean and uncertainty
        predictions = np.array(predictions)
        mean_prob = np.mean(predictions, axis=0)[0]
        uncertainty = np.std(predictions, axis=0)[0]
        
        predicted_idx = np.argmax(mean_prob)
        confidence = mean_prob[predicted_idx] * 100
        
        # Adjust confidence based on uncertainty
        adjusted_confidence = confidence * (1 - uncertainty[predicted_idx])
        
        return predicted_idx, adjusted_confidence, mean_prob
    
    def adaptive_threshold_prediction(self, probabilities: np.ndarray, 
                                    threshold_high: float = 0.75, 
                                    threshold_medium: float = 0.45,
                                    healthy_threshold: float = 0.30) -> str:
        """
        Adaptive confidence thresholding with healthy skin detection
        """
        max_prob = np.max(probabilities)
        
        # Check if all probabilities are low (indicating healthy skin)
        if max_prob < healthy_threshold:
            return "healthy"
        elif max_prob >= threshold_high:
            return "high"
        elif max_prob >= threshold_medium:
            return "medium"
        else:
            return "low"
    
    def advanced_prediction(self, tta_images: List[torch.Tensor]) -> dict:
        """
        Combined advanced prediction using ensemble + MC dropout + calibration
        """
        # Ensemble prediction
        ensemble_idx, ensemble_conf, ensemble_probs = self.ensemble_prediction(tta_images)
        
        # Monte Carlo dropout on first image
        mc_idx, mc_conf, mc_probs = self.monte_carlo_dropout(tta_images[0])
        
        # Combine predictions (weighted average)
        final_probs = 0.7 * ensemble_probs + 0.3 * mc_probs
        final_idx = np.argmax(final_probs)
        final_conf = final_probs[final_idx] * 100
        
        # Adaptive thresholding with healthy detection
        confidence_level = self.adaptive_threshold_prediction(final_probs)
        
        # Special handling for healthy skin
        is_healthy = confidence_level == "healthy"
        
        return {
            'predicted_class': int(final_idx) if not is_healthy else -1,
            'confidence': float(final_conf),
            'confidence_level': confidence_level,
            'probabilities': final_probs,
            'ensemble_confidence': float(ensemble_conf),
            'mc_confidence': float(mc_conf),
            'is_healthy': is_healthy
        }

def optimize_model_for_accuracy(model):
    """
    Apply model optimizations for better accuracy
    """
    # Apply optimizations
    model.eval()
    
    # Freeze batch norm for more stable inference
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False
    
    return model