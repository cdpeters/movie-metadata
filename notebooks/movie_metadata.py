import marimo

__generated_with = "0.6.19"
app = marimo.App(width="medium")


@app.cell
def __():
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs

    return Path, cs, mo, pl


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Movie Metadata Analysis
        ## Extract
        ---
        ### Read CSV
        `mm_raw.head()`:
        """
    )
    return


@app.cell
def __(Path, pl):
    movie_metadata_path = Path.cwd() / "imdb_top_1000.csv"

    # mm_raw = raw movie metadata dataframe. `Released_Year` is mapped to a string because
    # there is at least 1 value in that column that cannot be inferred as an integer. This
    # is investigated further below.
    mm_raw = pl.read_csv(
        source=movie_metadata_path, dtypes={"Released_Year": pl.String}
    )
    mm_raw.head()
    return mm_raw, movie_metadata_path


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Transform
        ---
        ### Structural/Metadata
        - `mm_pre` collects all transformations to the dataframe's stucture or metadata.

        `mm_pre.columns`:
        """
    )
    return


@app.cell
def __(mm_raw, pl):
    # Drop the `Poster_Link`, `Overview`, and `Certificate` columns.
    mm_pre = mm_raw.select(
        pl.col("*").exclude("Poster_Link", "Overview", "Certificate")
    )

    # Convert column names to lowercase.
    mm_pre = mm_pre.rename(lambda col_name: col_name.lower())

    # Column reordering.
    col_order = [
        "series_title",
        "released_year",
        "genre",
        "director",
        "star1",
        "star2",
        "star3",
        "star4",
        "runtime",
        "gross",
        "meta_score",
        "imdb_rating",
        "no_of_votes",
    ]
    mm_pre = mm_pre.select(col_order)

    mm_pre.columns
    return col_order, mm_pre


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Explore the Dataset
        #### Number of Columns Per Data Type
        """
    )
    return


@app.cell
def __(mm_pre, pl):
    pl.Series(mm_pre.dtypes).value_counts(sort=True)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        #### Column Names Per Data Type
        ##### Integer Columns
        """
    )
    return


@app.cell
def __(cs, mm_pre):
    mm_pre.select(cs.integer()).columns
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("##### String Columns")
    return


@app.cell
def __(cs, mm_pre):
    mm_pre.select(cs.string()).columns
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("`released_year`, `runtime`, and `gross` should all be integer columns.")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("##### Float Columns")
    return


@app.cell
def __(cs, mm_pre):
    mm_pre.select(cs.float()).columns
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("#### Find the `released_year` Values That Prevent Casting to Integer Column")
    return


@app.cell
def __(mm_pre, pl):
    mm_pre.filter(pl.col("released_year").cast(pl.Int64, strict=False).is_null())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        'The only problem row preventing casting is the movie `"Apollo 13"`. The release year for this movie is 1995 and will be updated in the **Column Transformations** section below.'
    )
    return


@app.cell
def __(mm_pre):
    mm_pre.null_count()
    return


@app.cell
def __(mo):
    mo.md(
        "Null values are only found in the `gross` and `meta_score` columns. This is not really an issue therefore no action will be taken for null values."
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Column Transformations
        1. `released_year`
            - There is one value that is preventing the column from being cast as an integer column. The movie is `"Apollo 13"`; the `released_year` value is `"PG"` and should be changed to `"1995"`. Then the column can be cast to an integer column. `pl.lit()` is needed because polars would otherwise try to look for a column named `"1995"`. Now the column can be cast to an integer type.
        1. `genre`
            - The values need to be split on ", ". The split operation will cast the column as a list type.
        1. `runtime`
            - The characters " min" need to be removed. Then cast to an integer type.
        1. `gross`
            - All of the "," characters need to be removed. Then cast to an integer type.

        `mm.head()`:
        """
    )
    return


@app.cell
def __(mm_pre, pl):
    mm = mm_pre.with_columns(
        # `released_year`.
        pl.when(pl.col("released_year") == "PG")
        .then(pl.lit("1995"))
        .otherwise(pl.col("released_year"))
        .cast(pl.Int16)
        .alias("released_year"),
        # `genre`, `runtime`, and `gross`.
        pl.col("genre").str.split(", "),
        pl.col("runtime").str.strip_chars_end(characters=" min").cast(pl.Int16),
        pl.col("gross")
        .str.replace_all(pattern=",", value="", literal=True)
        .cast(pl.Int32),
    )

    # Adjust max allowed string length in output.
    max_title_len = mm["series_title"].str.len_bytes().max()
    pl.Config.set_fmt_str_lengths(max_title_len)

    mm.head()
    return max_title_len, mm


@app.cell(hide_code=True)
def __(mo):
    mo.md("#### Grouping by `imdb_rating`")
    return


@app.cell
def __(mm, pl):
    mm.group_by(
        pl.col("imdb_rating")
        .cut(
            breaks=[8.0, 9.0],
            labels=["rating < 8", "8 <= rating < 9", "rating >= 9"],
            left_closed=True,
        )
        .alias("imdb_rating_group")
    ).agg(pl.col("imdb_rating").mean().alias("avg_rating_per_group").round(2))
    return


@app.cell
def __(mm):
    mm.describe()
    return


@app.cell
def __(mm):
    mm.head()
    return


@app.cell
def __(mm):
    mm.n_unique(subset="series_title")
    return


@app.cell
def __(mm, pl):
    mm.filter(pl.col("series_title").is_duplicated())
    return


@app.cell
def __(mm, pl):
    mm.filter(pl.struct("series_title", "released_year").is_duplicated())
    return


if __name__ == "__main__":
    app.run()
