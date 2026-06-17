"""Script to prepare the interventional dataset from LibriTTS.

Copyright (c) Jack Cox
SPDX-License-Identifier: MIT
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRITTS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEED = 42


def uuid(text: str) -> str:
    """Generate a deterministic id for a given text string.

    Args:
        text (str): The input text string.

    Returns:
        str: A deterministic UUID string based on the input text.

    """
    text_bytes = text.encode("utf-8")
    hash_object = hashlib.sha256(text_bytes)
    hex_dig = hash_object.hexdigest()
    return hex_dig[:12]  # Use the first 12 characters for a shorter ID


def libritts_to_dataframe(
    dataset: LIBRITTS,
    max_length: float = 10.0,
    min_length: float = 3.0,
) -> pd.DataFrame:
    """Structure LibriTTS dataset into a pandas DataFrame.

    Max and min length are in seconds, and are used to filter out very long or short
    audio samples.

    Args:
        dataset (LIBRITTS): The LibriTTS dataset.
        max_length (float): Maximum length of audio in seconds.
        min_length (float): Minimum length of audio in seconds.

    Returns:
        pd.DataFrame: A DataFrame with columns for speaker ID, audio tensor, transcript,
        and sample rate.

    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    data = []
    for item in dataloader:
        audio = item[0]
        sample_rate = item[1]
        transcript = item[3][0]
        speaker_id = int(item[4])
        # Filter by length in seconds
        length = audio.shape[-1] / sample_rate
        if length > max_length or length < min_length:
            continue
        data.append((speaker_id, audio, transcript, sample_rate))
    return pd.DataFrame(
        data,
        columns=["speaker_id", "audio", "transcript", "sample_rate"],
    )


def extract_libritts_transcripts(
    dataset: LIBRITTS,
    max_length: int = 15,
    min_length: int = 5,
) -> pd.Series:
    """Extract unique transcripts from LibriTTS.

    Removes duplicates and filters by length in words to ensure a high quality set of
    target transcripts.

    Args:
        dataset (LIBRITTS): The LibriTTS dataset.
        max_length (int): Maximum length of transcript in words.
        min_length (int): Minimum length of transcript in words.

    Returns:
        pd.Series: A Series of unique transcripts.

    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    transcripts = []
    for item in dataloader:
        transcript = item[3][0]
        transcripts.append(transcript)
    # Filter out duplicates and by length in words
    log.info("Extracted %d transcripts from LibriTTS dataset.", len(transcripts))
    transcripts_series = pd.Series(transcripts).drop_duplicates().reset_index(drop=True)
    log.info(
        "Filtered to %d unique transcripts after removing duplicates.",
        len(transcripts_series),
    )
    word_counts = transcripts_series.str.split().str.len()
    transcripts_series = transcripts_series[
        word_counts.between(min_length, max_length)
    ].reset_index(drop=True)
    log.info(
        "Filtered to %d unique transcripts with length between %d and %d words.",
        len(transcripts_series),
        min_length,
        max_length,
    )
    return transcripts_series


def construct_df(
    ref_df: pd.DataFrame,
    sentences: np.ndarray,
    ref_dir: Path,
) -> pd.DataFrame:
    """Construct the final DataFrame for the interventional dataset.

    Args:
        ref_df (pd.DataFrame): DataFrame containing reference audio and transcripts for
        each speaker.
        sentences (np.ndarray): Array of target transcripts to be paired with the
        reference audio.
        ref_dir (Path): Directory to save reference audio files.

    Returns:
        pd.DataFrame: The final DataFrame for the interventional dataset.

    """
    dataframe = []
    ref_dir.mkdir(parents=True, exist_ok=True)
    ref_by_speaker = ref_df.groupby("speaker_id")
    n_speakers = len(ref_by_speaker)
    for speaker_id, group in ref_by_speaker:
        ref_audios = group["audio"].to_list()
        ref_transcripts = group["transcript"].to_list()
        sample_rate = group["sample_rate"].to_list()
        # Save reference audios
        for i, ref_audio in enumerate(ref_audios):
            sentence_id = uuid(ref_transcripts[i])
            ref_audio_path = ref_dir / f"speaker_{speaker_id}_ref_{sentence_id}.wav"
            if not ref_audio_path.exists():
                torchaudio.save(ref_audio_path, ref_audio.squeeze(0), sample_rate[i])
            # Add to dataframe
            dataframe.append({
                "ref_wav": ref_audio_path,
                "ref_text": ref_transcripts[i],
                "speaker_id": speaker_id,
            })
    dataframe = pd.DataFrame(dataframe)

    # Tile transcripts
    transcript_col = np.tile(sentences, n_speakers)
    dataframe = dataframe.assign(target_text=transcript_col)
    dataframe["path"] = (
        "speaker_"
        + dataframe["speaker_id"].astype(str)
        + "_target_"
        + dataframe["target_text"].apply(uuid)
        + ".wav"
    )

    return dataframe


def prepare_train_set(
    libritts_df: pd.DataFrame,
    sentences: pd.Series,
    ref_dir: Path,
    n_sentences: int = 256,
    n_speakers: int = 32,
) -> pd.DataFrame:
    """Prepare a training set from LibriTTS seed speakers and transcripts.

    Args:
        libritts_df (pd.DataFrame): LibriTTS dataframe.
        sentences (pd.Series): Series of target transcripts.
        ref_dir (Path): Directory to save reference audio files.
        n_sentences (int): Number of unique target sentences to sample.
        n_speakers (int): Number of unique speakers to sample.

    Returns:
        pd.DataFrame: The final DataFrame for the interventional dataset.

    """
    speakers = pd.Series(libritts_df["speaker_id"].unique())
    # Sample unique speakers if there are more than n_speakers available
    if n_speakers > len(speakers):
        n_speakers = len(speakers)
        log.warning(
            "Only %d unique speakers available. Using all available speakers.",
            n_speakers,
        )
    sampled_speakers = speakers.sample(n=n_speakers, random_state=SEED)
    libritts_df = libritts_df[
        libritts_df["speaker_id"].isin(sampled_speakers)
    ].reset_index(drop=True)
    # For each speaker, sample n_sentences ref_audio examples
    ref_df = libritts_df.groupby("speaker_id").sample(
        n=n_sentences,
        replace=True,
        random_state=SEED,
    )
    log.info(
        "Prepared dataset of reference audio with %d speakers and %d total samples.",
        len(sampled_speakers),
        len(ref_df),
    )

    # Sample sentences for target transcripts
    transcripts = sentences.sample(n=n_sentences, random_state=SEED).to_numpy()
    log.info("Prepared %d unique target transcripts for generation.", len(transcripts))

    # Construct final dataframe
    return construct_df(ref_df, transcripts, ref_dir)


def prepare_val_set(
    train_dataframe: pd.DataFrame,
    libritts_df: pd.DataFrame,
    sentences: pd.Series,
    ref_dir: Path,
    n_speakers: int = 6,
) -> pd.DataFrame:
    """Prepare a validation set from unseen speakers and transcripts.

    Args:
        train_dataframe (pd.DataFrame): The training set dataframe.
        libritts_df (pd.DataFrame): The LibriTTS dataframe.
        sentences (pd.Series): Series of target transcripts.
        ref_dir (Path): Directory to save reference audio files.
        n_speakers (int): Number of unique speakers to sample for the val set.

    Returns:
        pd.DataFrame: The final DataFrame for the validation set of the interventional
        dataset.

    """
    n_sentences = train_dataframe["target_text"].nunique()
    speakers = pd.Series(libritts_df["speaker_id"].unique())
    # Remove speakers from training set
    train_speakers = pd.Series(train_dataframe["speaker_id"].unique())
    available_speakers = speakers[~speakers.isin(train_speakers)]
    # Sample unique speakers for val set
    if n_speakers > len(available_speakers):
        n_speakers = len(available_speakers)
        log.warning(
            "Only %d unique speakers available for val set.",
            n_speakers,
        )
    sampled_speakers = available_speakers.sample(n=n_speakers, random_state=SEED)
    val_ref_df = libritts_df[
        libritts_df["speaker_id"].isin(sampled_speakers)
    ].reset_index(drop=True)
    # For each speaker, sample n_sentences ref_audio examples
    val_ref_df = val_ref_df.groupby("speaker_id").sample(
        n=n_sentences,
        replace=True,
        random_state=SEED,
    )
    log.info(
        "Prepared val set of reference audio with %d speakers and %d total samples.",
        len(sampled_speakers),
        len(val_ref_df),
    )

    # Sample sentences for target transcripts which are not in training set
    train_sentences = pd.Series(train_dataframe["target_text"].unique())
    available_sentences = sentences[~sentences.isin(train_sentences)]
    val_transcripts = available_sentences.sample(
        n=n_sentences,
        random_state=42,
    ).to_numpy()
    log.info(
        "Prepared %d unique target transcripts for val generation.",
        len(val_transcripts),
    )

    # Construct final val dataframe
    return construct_df(val_ref_df, val_transcripts, ref_dir)


if __name__ == "__main__":
    # Check torchaudio backend
    log.info("Available torchaudio backends: %s", torchaudio.list_audio_backends())
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True, parents=True)
    output_dir = data_dir / "synthetic"
    ref_dir = output_dir / "reference_audio"
    max_speakers = 32
    max_sentences = 256

    # Check for dataset metadata
    train_set_path = output_dir / "train_set.tsv"
    val_set_path = output_dir / "val_set.tsv"
    if not train_set_path.exists() or not val_set_path.exists():
        # Load LibriTTS dataset
        log.info("Loading LibriTTS dataset...")
        libritts_dataset = LIBRITTS(
            data_dir,
            url="test-clean",
            download=True,
        )
        libritts_df = libritts_to_dataframe(
            libritts_dataset,
            max_length=10.0,
            min_length=3.0,
        )
        # Extract transcripts
        log.info("Extracting transcripts...")
        transcripts = extract_libritts_transcripts(
            libritts_dataset,
            max_length=15,
            min_length=5,
        )
        # Train set
        if not train_set_path.exists():
            log.info("No existing dataset. Preparing dataset metadata...")
            output_dir.mkdir(parents=True, exist_ok=True)
            train_dataframe = prepare_train_set(
                libritts_df,
                transcripts,
                ref_dir,
                n_sentences=max_sentences,
                n_speakers=max_speakers,
            )
            train_dataframe.to_csv(f"{output_dir}/train_set.tsv", sep="\t", index=False)
        else:
            # Load existing dataset metadata
            log.info("Loading existing dataset metadata...")
            train_dataframe = pd.read_csv(f"{output_dir}/train_set.tsv", sep="\t")

        # Val set
        log.info("Preparing val set metadata...")
        val_dataframe = prepare_val_set(
            train_dataframe,
            libritts_df,
            transcripts,
            ref_dir,
            n_speakers=6,
        )
        val_dataframe.to_csv(f"{output_dir}/val_set.tsv", sep="\t", index=False)
    else:
        log.info("Loading existing dataset metadata...")
        train_dataframe = pd.read_csv(f"{output_dir}/train_set.tsv", sep="\t")
        val_dataframe = pd.read_csv(f"{output_dir}/val_set.tsv", sep="\t")
