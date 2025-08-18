import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns
    from plotly.subplots import make_subplots

    pl.Config.set_tbl_rows(25)
    return Path, cs, mo, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Movie Metadata Analysis
    ## Extract
    1. Read the CSV into a dataframe.
        - `mm_raw` - raw movie metadata dataframe
        - `mm` - transformed movie metadata dataframe
        - `Released_Year` is mapped to a string because there is at least 1 value in that column that cannot be inferred as an integer
        - this is investigated in additional cells below
    2. Remove columns that will not be used in the analysis:
        - `Poster_Link`
        - `Overview`
        - `Certificate`
    3. Convert all column names to lowercase for uniformity.
    4. Adjust the max allowed length of strings in outputs to easily see the full titles of movies
        - the movie title is treated as the priority for viewing string output in the dataframe
    """
    )
    return


@app.cell
def _(Path, pl):
    movie_metadata_path: Path = Path.cwd() / "imdb_top_1000.csv"

    mm_raw = pl.read_csv(
        source=movie_metadata_path, schema_overrides={"Released_Year": pl.String}
    )

    # Drop the columns mentioned above and rename columns as all lowercase.
    mm = mm_raw.select(
        pl.col("*").exclude("Poster_Link", "Overview", "Certificate")
    ).rename(lambda col_name: col_name.lower())

    # Change the display so full titles are shown.
    max_title_len = mm["series_title"].str.len_bytes().max()
    pl.Config.set_fmt_str_lengths(max_title_len)

    mm
    return (mm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Explore
    ### Columns Per Data Type
    """
    )
    return


@app.cell
def _(mm, pl):
    # First check what data types are in the dataframe.
    pl.Series(mm.dtypes).value_counts(sort=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Integer Columns""")
    return


@app.cell
def _(cs, mm):
    mm.select(cs.integer()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### String Columns""")
    return


@app.cell
def _(cs, mm):
    mm.select(cs.string()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""`released_year`, `runtime`, and `gross` should all be integer columns.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Float Columns""")
    return


@app.cell
def _(cs, mm):
    mm.select(cs.float()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Find the `released_year` Values That Prevent Casting to Integer Column""")
    return


@app.cell
def _(mm, pl):
    # Cast to integer by supressing the error that results. Then find the null values.
    mm.filter(pl.col("released_year").cast(pl.Int64, strict=False).is_null())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The only problem row preventing casting is the movie `"Apollo 13"`. The release year for this movie is `1995` and will be updated in the **Column Transformations** section below."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Missing/Null Values""")
    return


@app.cell
def _(mm):
    mm.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Null values are only found in the `gross` and `meta_score` columns. This is not really an issue therefore no action will be taken for null values."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Duplicates
    #### `series_title` Duplicates
    """
    )
    return


@app.cell
def _(mm, pl):
    mm.filter(pl.col("series_title").is_duplicated())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""There are two movies that share the same title but are clearly different movies. No action is needed."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""#### Combination of `series_title` and `released_year` Duplicates""")
    return


@app.cell
def _(mm, pl):
    # `pl.struct` is used to tie the two columns together in one object.
    mm.filter(pl.struct("series_title", "released_year").is_duplicated())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""There are no duplicates when `series_title` and `released_year` are used in combination as a unique identifier for each row. Although not the goal of the project, if a database was constructed, `series_title` with `released_year` could be used as a composite primary key."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Transformations
    ### Column Reordering
    """
    )
    return


@app.cell
def _(mm):
    # A more logical ordering.
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

    mm_transformed = mm.select(col_order)
    mm_transformed.head()
    return (mm_transformed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Column Transformations
    1. `released_year`
        - There is one value that is preventing the column from being cast as an integer column. The movie is `"Apollo 13"`; the `released_year` value is `"PG"` and should be changed to `"1995"`. Then the column can be cast to an integer column. `pl.lit()` is needed because polars would otherwise try to look for a column named `"1995"`. Now the column can be cast to an integer type.
    1. `genre`
        - The values need to be split on ", ". The split operation will cast the column as a list type.
    1. `runtime`
        - The characters " min" need to be removed. Then cast to an integer type.
    1. `gross`
        - All of the "," characters need to be removed. Then cast to an integer type.
    """
    )
    return


@app.cell
def _(mm_transformed, pl):
    mm = mm_transformed.with_columns(
        # Convert `released_year` to integer column.
        pl.when(pl.col("released_year") == "PG")
        .then(pl.lit("1995"))
        .otherwise(pl.col("released_year"))
        .cast(pl.Int16)
        .alias("released_year"),
        # Convert `genre` to a list Column.
        pl.col("genre").str.split(", "),
        # Address the `runtime` column.
        pl.col("runtime").str.strip_chars_end(characters=" min").cast(pl.Int16),
        # Address the `gross` column.
        pl.col("gross").str.replace_all(pattern=",", value="", literal=True).cast(pl.Int32),
    )
    mm.head()
    return (mm,)


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
