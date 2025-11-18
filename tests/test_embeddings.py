"""Unit tests for EmbeddingModel class."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from parakeet_search.embeddings import EmbeddingModel


class TestEmbeddingModelInit:
    """Test EmbeddingModel initialization."""

    def test_init_default_model(self):
        """Test initialization with default model."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            em = EmbeddingModel()

            assert em.model_name == "all-MiniLM-L6-v2"
            assert em.embedding_dim == 384
            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    def test_init_custom_model(self):
        """Test initialization with custom model."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            em = EmbeddingModel(model_name="all-mpnet-base-v2")

            assert em.model_name == "all-mpnet-base-v2"
            assert em.embedding_dim == 768
            mock_st.assert_called_once_with("all-mpnet-base-v2")

    def test_init_stores_model_reference(self):
        """Test that init properly stores model reference."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            em = EmbeddingModel()

            assert em.model is mock_model


class TestEmbeddingModelEmbedText:
    """Test EmbeddingModel.embed_text() method."""

    @pytest.fixture
    def embedding_model(self):
        """Create a mock EmbeddingModel for testing."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            em = EmbeddingModel()
            yield em

    def test_embed_text_returns_numpy_array(self, embedding_model):
        """Test that embed_text returns a numpy array."""
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text("hello world")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_embed_text_calls_model_encode(self, embedding_model):
        """Test that embed_text calls model.encode with correct parameters."""
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        embedding_model.embed_text("test text")

        embedding_model.model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    def test_embed_text_single_character(self, embedding_model):
        """Test embedding of single character."""
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text("a")

        assert isinstance(result, np.ndarray)
        embedding_model.model.encode.assert_called_once()

    def test_embed_text_long_text(self, embedding_model):
        """Test embedding of long text."""
        long_text = "word " * 1000  # 5000 words
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text(long_text)

        assert isinstance(result, np.ndarray)

    def test_embed_text_special_characters(self, embedding_model):
        """Test embedding of text with special characters."""
        special_text = "Hello! @#$% & 世界 🌍"
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text(special_text)

        assert isinstance(result, np.ndarray)
        embedding_model.model.encode.assert_called_once_with(special_text, convert_to_numpy=True)

    def test_embed_text_empty_string(self, embedding_model):
        """Test embedding of empty string."""
        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text("")

        assert isinstance(result, np.ndarray)


class TestEmbeddingModelEmbedTexts:
    """Test EmbeddingModel.embed_texts() method."""

    @pytest.fixture
    def embedding_model(self):
        """Create a mock EmbeddingModel for testing."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            em = EmbeddingModel()
            yield em

    def test_embed_texts_returns_2d_array(self, embedding_model):
        """Test that embed_texts returns a 2D numpy array."""
        mock_embeddings = np.random.randn(3, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        texts = ["text1", "text2", "text3"]
        result = embedding_model.embed_texts(texts)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 384)

    def test_embed_texts_calls_model_encode(self, embedding_model):
        """Test that embed_texts calls model.encode with correct parameters."""
        mock_embeddings = np.random.randn(2, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        texts = ["text1", "text2"]
        embedding_model.embed_texts(texts)

        embedding_model.model.encode.assert_called_once_with(
            texts, convert_to_numpy=True, show_progress_bar=True
        )

    def test_embed_texts_single_text(self, embedding_model):
        """Test embedding of single text in list."""
        mock_embeddings = np.random.randn(1, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        result = embedding_model.embed_texts(["single text"])

        assert result.shape == (1, 384)

    def test_embed_texts_many_texts(self, embedding_model):
        """Test embedding of many texts."""
        num_texts = 100
        mock_embeddings = np.random.randn(num_texts, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        texts = [f"text {i}" for i in range(num_texts)]
        result = embedding_model.embed_texts(texts)

        assert result.shape == (num_texts, 384)

    def test_embed_texts_empty_list(self, embedding_model):
        """Test embedding of empty list."""
        mock_embeddings = np.array([]).reshape(0, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        result = embedding_model.embed_texts([])

        assert result.shape == (0, 384)

    def test_embed_texts_mixed_lengths(self, embedding_model):
        """Test embedding of texts with different lengths."""
        mock_embeddings = np.random.randn(3, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        texts = [
            "short",
            "a medium length text here",
            "this is a much longer text that contains more words and more information " * 10,
        ]
        result = embedding_model.embed_texts(texts)

        assert result.shape == (3, 384)

    def test_embed_texts_preserves_order(self, embedding_model):
        """Test that embed_texts preserves text order."""
        # Create distinct embeddings for each text
        embeddings = [
            np.full(384, 0.1),
            np.full(384, 0.2),
            np.full(384, 0.3),
        ]
        embedding_model.model.encode.return_value = np.array(embeddings)

        texts = ["text1", "text2", "text3"]
        result = embedding_model.embed_texts(texts)

        # Verify order is preserved by checking the embeddings
        assert np.allclose(result[0], embeddings[0])
        assert np.allclose(result[1], embeddings[1])
        assert np.allclose(result[2], embeddings[2])


class TestEmbeddingModelIntegration:
    """Integration tests for EmbeddingModel."""

    def test_embedding_dimension_consistency(self):
        """Test that embedding dimension is consistent across calls."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            em = EmbeddingModel()

            # Embed single text
            mock_model.encode.return_value = np.random.randn(384)
            single_result = em.embed_text("test")
            assert single_result.shape == (384,)

            # Embed multiple texts
            mock_model.encode.return_value = np.random.randn(5, 384)
            batch_result = em.embed_texts(["test"] * 5)
            assert batch_result.shape == (5, 384)

    def test_model_name_attribute(self):
        """Test that model_name attribute is preserved."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_st.return_value = mock_model

            em = EmbeddingModel(model_name="custom-model")

            assert em.model_name == "custom-model"
