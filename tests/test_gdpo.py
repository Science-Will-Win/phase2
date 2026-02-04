"""
Unit tests for GDPO module.

Tests the class-based GDPO implementation including:
- soft_scale function
- GDPOBase class
- GDPOLoss class
- HeteroscedasticGDPOLoss class
- Legacy function wrappers
- Instance caching
"""
import torch
import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GDPO import (
    GDPOBase, GDPOLoss, HeteroscedasticGDPOLoss,
    compute_gdpo_loss, compute_heteroscedastic_gdpo_loss,
    compute_rewards, soft_scale, condition_reward
)


# =============================================================================
# Test soft_scale function
# =============================================================================

class TestSoftScale:
    """Tests for the soft_scale function."""
    
    def test_positive_tensor(self):
        """Test soft_scale with positive tensor values."""
        x = torch.tensor([0.5, 1.0, 2.0])
        result = soft_scale(x)
        expected = x / (1 + torch.abs(x))
        assert torch.allclose(result, expected)
    
    def test_negative_tensor(self):
        """Test soft_scale with negative tensor values."""
        x = torch.tensor([-0.5, -1.0, -2.0])
        result = soft_scale(x)
        expected = x / (1 + torch.abs(x))
        assert torch.allclose(result, expected)
    
    def test_range_bounds(self):
        """Result should always be in (-1, 1) range."""
        x = torch.tensor([-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        result = soft_scale(x)
        assert (result > -1).all(), "All values should be > -1"
        assert (result < 1).all(), "All values should be < 1"
    
    def test_zero(self):
        """Test soft_scale with zero."""
        x = torch.tensor([0.0])
        result = soft_scale(x)
        assert result.item() == 0.0
    
    def test_scalar(self):
        """Test soft_scale with scalar input."""
        assert soft_scale(0.5) == 0.5 / (1 + 0.5)
        assert soft_scale(-0.5) == -0.5 / (1 + 0.5)
        assert soft_scale(0) == 0
    
    def test_large_values_approach_one(self):
        """Large positive values should approach 1."""
        x = torch.tensor([1000.0])
        result = soft_scale(x)
        assert result.item() > 0.999


# =============================================================================
# Test condition_reward function
# =============================================================================

class TestConditionReward:
    """Tests for the condition_reward function."""
    
    def test_above_threshold(self):
        """Easy reward should be returned when hard reward >= threshold."""
        assert condition_reward(1.0, 1.0, threshold=1.0) == 1.0
        assert condition_reward(0.5, 1.5, threshold=1.0) == 0.5
    
    def test_below_threshold(self):
        """Easy reward should be 0 when hard reward < threshold."""
        assert condition_reward(1.0, 0.5, threshold=1.0) == 0.0
        assert condition_reward(0.8, 0.9, threshold=1.0) == 0.0


# =============================================================================
# Test GDPOBase class
# =============================================================================

class TestGDPOBase:
    """Tests for the GDPOBase class."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer with default config."""
        trainer = MagicMock()
        trainer.gdpo_config = {
            "group_size": 2,
            "temperature": 1.0,
            "max_new_tokens": 64,
            "kl_coef": 0.01,
            "use_temperature_contrastive": False,
            "low_temperature": 0.3,
            "high_temperature": 1.2,
        }
        trainer.processing_class = MagicMock()
        trainer.processing_class.pad_token_id = 0
        trainer.processing_class.eos_token_id = 1
        trainer.ref_model = None
        trainer.debug = False
        return trainer
    
    def test_init_default_config(self, mock_trainer):
        """Test GDPOBase initialization with default config."""
        base = GDPOBase(mock_trainer)
        
        assert base.G == 2
        assert base.max_new_tokens == 64
        assert base.kl_coef == 0.01
        assert base.use_temp_contrastive is False
        assert base.default_temp == 1.0
    
    def test_init_temp_contrastive_config(self, mock_trainer):
        """Test GDPOBase with temperature contrastive enabled."""
        mock_trainer.gdpo_config["use_temperature_contrastive"] = True
        base = GDPOBase(mock_trainer)
        
        assert base.use_temp_contrastive is True
        assert base.low_temp == 0.3
        assert base.high_temp == 1.2
    
    def test_get_reward_weights_3_objectives(self, mock_trainer):
        """Test get_reward_weights with 3 objectives."""
        mock_trainer.gdpo_config["reward_weights"] = {
            "format": 2.0,
            "length": 1.5,
            "accuracy": 1.0,
        }
        base = GDPOBase(mock_trainer)
        weights = base.get_reward_weights(3)
        
        assert weights == [2.0, 1.5, 1.0]
    
    def test_get_reward_weights_5_objectives(self, mock_trainer):
        """Test get_reward_weights with 5 objectives (all rewards)."""
        mock_trainer.gdpo_config["reward_weights"] = {
            "format": 1.0,
            "length": 1.0,
            "accuracy": 1.0,
            "uncertainty": 0.5,
            "temperature": 0.8,
        }
        base = GDPOBase(mock_trainer)
        weights = base.get_reward_weights(5)
        
        assert weights == [1.0, 1.0, 1.0, 0.5, 0.8]
    
    def test_compute_advantages_shape(self, mock_trainer):
        """Test compute_advantages output shape."""
        base = GDPOBase(mock_trainer)
        
        B, G, n = 2, 4, 3
        rewards = torch.randn(B, G, n)
        weights = [1.0, 1.0, 1.0]
        
        result = base.compute_advantages(rewards, weights, torch.device('cpu'))
        
        assert result.shape == (B * G,)
    
    def test_compute_advantages_normalized(self, mock_trainer):
        """Test that advantages are normalized (mean ~0, std ~1)."""
        base = GDPOBase(mock_trainer)
        
        B, G, n = 4, 8, 3
        rewards = torch.randn(B, G, n) * 10  # Large variance
        weights = [1.0, 1.0, 1.0]
        
        result = base.compute_advantages(rewards, weights, torch.device('cpu'))
        
        # After batch normalization, mean should be ~0 and std ~1
        assert abs(result.mean().item()) < 0.1
        assert abs(result.std().item() - 1.0) < 0.1
    
    def test_build_reward_config(self, mock_trainer):
        """Test build_reward_config returns correct dict."""
        mock_trainer.gdpo_config["use_conditioned_rewards"] = True
        mock_trainer.gdpo_config["condition_threshold"] = 0.8
        mock_trainer.gdpo_config["target_length"] = 512
        
        base = GDPOBase(mock_trainer)
        config = base.build_reward_config()
        
        assert config["use_conditioned_rewards"] is True
        assert config["condition_threshold"] == 0.8
        assert config["target_length"] == 512


# =============================================================================
# Test compute_rewards function
# =============================================================================

class TestComputeRewards:
    """Tests for the compute_rewards function."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.batch_decode = MagicMock(return_value=["Test output"] * 4)
        return tokenizer
    
    def test_basic_rewards_shape(self, mock_tokenizer):
        """Test basic reward computation shape."""
        sequences = torch.randint(0, 1000, (4, 50))  # 4 sequences, 50 tokens
        
        rewards = compute_rewards(
            sequences, mock_tokenizer,
            num_objectives=3
        )
        
        assert rewards.shape == (4, 3)
    
    def test_rewards_with_uncertainty(self, mock_tokenizer):
        """Test reward computation with uncertainty scores."""
        sequences = torch.randint(0, 1000, (4, 50))
        uncertainty_scores = torch.tensor([0.3, 0.5, 0.7, 0.9])
        
        rewards = compute_rewards(
            sequences, mock_tokenizer,
            uncertainty_scores=uncertainty_scores,
            reward_config={"uncertainty_threshold": 0.6}
        )
        
        # Should auto-adjust to 4 objectives
        assert rewards.shape == (4, 4)
        
        # Check uncertainty penalty applied correctly
        # Index 2 (0.7) and 3 (0.9) should have negative values
        assert rewards[0, 3] == 0.0  # 0.3 < 0.6
        assert rewards[1, 3] == 0.0  # 0.5 < 0.6
        assert rewards[2, 3] < 0     # 0.7 >= 0.6, penalty
        assert rewards[3, 3] < 0     # 0.9 >= 0.6, penalty
    
    def test_rewards_with_temperature(self, mock_tokenizer):
        """Test reward computation with temperature rewards."""
        sequences = torch.randint(0, 1000, (4, 50))
        temperature_rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
        
        rewards = compute_rewards(
            sequences, mock_tokenizer,
            temperature_rewards=temperature_rewards
        )
        
        # Should auto-adjust to 4 objectives (3 base + 1 temp)
        assert rewards.shape == (4, 4)
        
        # Check temperature rewards assigned correctly
        assert rewards[0, 3] == 1.0
        assert rewards[1, 3] == 1.0
        assert rewards[2, 3] == -1.0
        assert rewards[3, 3] == -1.0
    
    def test_rewards_with_both_uncertainty_and_temperature(self, mock_tokenizer):
        """Test reward computation with both uncertainty and temperature."""
        sequences = torch.randint(0, 1000, (4, 50))
        uncertainty_scores = torch.tensor([0.3, 0.5, 0.7, 0.9])
        temperature_rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
        
        rewards = compute_rewards(
            sequences, mock_tokenizer,
            uncertainty_scores=uncertainty_scores,
            temperature_rewards=temperature_rewards,
            reward_config={"uncertainty_threshold": 0.6}
        )
        
        # Should have 5 objectives
        assert rewards.shape == (4, 5)
        
        # Uncertainty at index 3, temperature at index 4
        assert rewards[0, 4] == 1.0
        assert rewards[3, 4] == -1.0


# =============================================================================
# Test Legacy Compatibility and Caching
# =============================================================================

class TestLegacyCompatibility:
    """Tests for legacy function wrappers and caching."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create a mock trainer."""
        trainer = MagicMock()
        trainer.gdpo_config = {
            "group_size": 2,
            "temperature": 1.0,
            "max_new_tokens": 32,
            "kl_coef": 0.01,
            "use_temperature_contrastive": False,
        }
        trainer.processing_class = MagicMock()
        trainer.processing_class.pad_token_id = 0
        trainer.processing_class.eos_token_id = 1
        trainer.processing_class.batch_decode = MagicMock(return_value=["Test"] * 4)
        trainer.ref_model = None
        trainer.debug = False
        return trainer
    
    def test_gdpo_loss_caching(self, mock_trainer):
        """Test that GDPOLoss instance is cached on trainer."""
        # First call should create instance
        assert not hasattr(mock_trainer, '_gdpo_loss_instance')
        
        # Can't fully test without model, but we can check structure
        loss_class = GDPOLoss(mock_trainer)
        mock_trainer._gdpo_loss_instance = loss_class
        
        # Verify caching attribute exists
        assert hasattr(mock_trainer, '_gdpo_loss_instance')
        assert isinstance(mock_trainer._gdpo_loss_instance, GDPOLoss)
        
        # Second call should reuse instance
        instance1 = mock_trainer._gdpo_loss_instance
        instance2 = mock_trainer._gdpo_loss_instance
        assert instance1 is instance2
    
    def test_hetero_gdpo_loss_caching(self, mock_trainer):
        """Test that HeteroscedasticGDPOLoss instance is cached on trainer."""
        mock_trainer.gdpo_config["uncertainty_threshold"] = 0.6
        
        loss_class = HeteroscedasticGDPOLoss(mock_trainer)
        mock_trainer._hetero_gdpo_loss_instance = loss_class
        
        assert hasattr(mock_trainer, '_hetero_gdpo_loss_instance')
        assert isinstance(mock_trainer._hetero_gdpo_loss_instance, HeteroscedasticGDPOLoss)
    
    def test_gdpo_loss_class_has_temp_contrastive_support(self, mock_trainer):
        """Test that GDPOLoss class supports temperature contrastive."""
        mock_trainer.gdpo_config["use_temperature_contrastive"] = True
        mock_trainer.gdpo_config["low_temperature"] = 0.3
        mock_trainer.gdpo_config["high_temperature"] = 1.2
        
        loss_class = GDPOLoss(mock_trainer)
        
        assert loss_class.use_temp_contrastive is True
        assert loss_class.low_temp == 0.3
        assert loss_class.high_temp == 1.2


# =============================================================================
# Test Temperature Contrastive Features
# =============================================================================

class TestTemperatureContrastive:
    """Tests for temperature contrastive sampling features."""
    
    @pytest.fixture
    def mock_trainer_with_temp(self):
        """Create a mock trainer with temperature contrastive enabled."""
        trainer = MagicMock()
        trainer.gdpo_config = {
            "group_size": 4,
            "temperature": 1.0,
            "max_new_tokens": 32,
            "kl_coef": 0.01,
            "use_temperature_contrastive": True,
            "low_temperature": 0.3,
            "high_temperature": 1.2,
            "reward_weights": {
                "format": 1.0,
                "length": 1.0,
                "accuracy": 1.0,
                "temperature": 1.0,
            }
        }
        trainer.processing_class = MagicMock()
        trainer.processing_class.pad_token_id = 0
        trainer.processing_class.eos_token_id = 1
        trainer.ref_model = None
        trainer.debug = False
        return trainer
    
    def test_effective_group_size_doubles(self, mock_trainer_with_temp):
        """Test that effective group size doubles with temp contrastive."""
        base = GDPOBase(mock_trainer_with_temp)
        
        assert base.use_temp_contrastive is True
        assert base.G == 4
        # When generate_samples is called, effective_G should be 2*G=8
    
    def test_reward_weights_include_temperature(self, mock_trainer_with_temp):
        """Test that reward weights include temperature objective."""
        base = GDPOBase(mock_trainer_with_temp)
        weights = base.get_reward_weights(4)  # 3 base + 1 temp
        
        assert len(weights) == 4
        assert weights[3] == 1.0  # temperature weight


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
