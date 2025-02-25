from flytekit import task, workflow, ImageSpec

import os
import polars as pl
import s3fs

from io import StringIO
from enum import StrEnum
import polars as pl
import httpx


class CorpusUrlType(StrEnum):
    ALBUM = "https://github.com/sagesolar/Corpus-of-Taylor-Swift/raw/refs/heads/main/lyrics/album-song-lyrics.json"
    FLAT = "https://github.com/sagesolar/Corpus-of-Taylor-Swift/raw/refs/heads/main/tsv/cots-lyric-details.tsv"


class MetadataUrlType(StrEnum):
    ALBUM = "https://github.com/sagesolar/Corpus-of-Taylor-Swift/raw/refs/heads/main/tsv/cots-album-details.tsv"
    SONG = "https://github.com/sagesolar/Corpus-of-Taylor-Swift/raw/refs/heads/main/tsv/cots-song-details.tsv"
    WORD = "https://github.com/sagesolar/Corpus-of-Taylor-Swift/raw/refs/heads/main/tsv/cots-word-details.tsv"


def get_lyrics() -> pl.DataFrame:
    url = CorpusUrlType.FLAT
    r = httpx.get(url, follow_redirects=True)
    if r.status_code != 200:
        raise ValueError("Could not retrieve lyrics: %s", r.text)
    return (
        pl.read_csv(
            StringIO(r.text),
            separator="\t",
            has_header=False,
            new_columns=["Code", "Lyric"],
        )
        # Code = Album Code : Track Number : Lyric Line Number : Song Structure Part
        .with_columns(
            pl.col("Code")
            .str.split_exact(":", 3)
            .struct.rename_fields(
                [
                    "Album Code",
                    "Track Number",
                    "Lyric Line Number",
                    "Song Structure Part",
                ]
            )
            .alias("fields")
        ).unnest("fields")
        # Cast numbers to int
        .with_columns(
            pl.col("Track Number").cast(pl.Int64),
            pl.col("Lyric Line Number").cast(pl.Int64),
        )
    )


def get_metadata(metadata="album") -> pl.DataFrame:
    url = MetadataUrlType[metadata.upper()]
    r = httpx.get(url, follow_redirects=True)
    if r.status_code != 200:
        raise ValueError("Could not retrieve metadata: %s", r.text)
    return pl.read_csv(StringIO(r.text), separator="\t")


TSW_IMAGE_SPEC = ImageSpec(
    name="flyte-tsw",
    registry="ghcr.io/solanyn",
    packages=["httpx", "polars", "pyarrow", "s3fs", "flytekitplugins-polars"],
    python_version="3.11",
)


@task(container_image=TSW_IMAGE_SPEC)
def collect_lyrics() -> pl.DataFrame:
    return get_lyrics()


@task(container_image=TSW_IMAGE_SPEC)
def collect_album_metadata() -> pl.DataFrame:
    return get_metadata("album")


@task(container_image=TSW_IMAGE_SPEC)
def collect_song_metadata() -> pl.DataFrame:
    return get_metadata("song")


@task(container_image=TSW_IMAGE_SPEC)
def build_album_corpus(
    lyrics_df: pl.DataFrame,
    song_df: pl.DataFrame,
    album_df: pl.DataFrame,
) -> pl.DataFrame:
    return lyrics_df.join(
        album_df.select("Code", "Title").rename({"Title": "Album Title"}),
        left_on="Album Code",
        right_on="Code",
        how="inner",
    ).join(
        song_df.select("Album", "Track", "Title").rename({"Title": "Track Title"}),
        left_on=["Album Code", "Track Number"],
        right_on=["Album", "Track"],
    )


@task(container_image=TSW_IMAGE_SPEC)
def save_corpus(corpus_df: pl.DataFrame) -> None:
    s3 = s3fs.S3FileSystem(
        key=os.getenv("TSW_ACCESS_KEY_ID"),
        secret=os.getenv("TSW_SECRET_ACCESS_KEY"),
        endpoint_url=os.getenv("TSW_ENDPOINT_URL"),
    )

    for album_code in corpus_df.get_column("Album Code").unique().to_list():
        with s3.open(f"s3://tsw/{album_code}.parquet") as f:
            corpus_df.write_parquet(f)


@workflow
def collect_data():
    album_df = collect_album_metadata()
    song_df = collect_song_metadata()
    lyrics_df = collect_lyrics()
    corpus_df = build_album_corpus(
        lyrics_df=lyrics_df, song_df=song_df, album_df=album_df
    )
    save_corpus(corpus_df=corpus_df)


if __name__ == "__main__":
    collect_data()
