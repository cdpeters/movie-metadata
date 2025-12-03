import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium", auto_download=["html", "ipynb"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Movie Metadata Analysis
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns
    from great_tables import GT
    from plotly.subplots import make_subplots
    from itertools import product

    pl.Config.set_tbl_rows(25)
    return Path, cs, go, make_subplots, mo, np, pl, product, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract
    ### Main variables
    | variable         | description                                                                           |
    | ---------------- | ------------------------------------------------------------------------------------- |
    | `mm_raw`         | raw movie metadata dataframe                                                          |
    | `mm_transformed` | intermediate dataframe used to explore the columns for more necessary transformations |
    | `mm`             | final movie metadata dataframe with all transformations applied, ready for analysis   |

    ### Steps/Notes
    1. Read CSV into a dataframe.
        - `Released_Year` is mapped to a string because there is at least 1 value in that column that cannot be inferred as an integer. This is investigated in additional cells below.
    2. Remove columns that will not be used in the analysis:
        - `Poster_Link`
        - `Overview`
        - `Certificate`
    3. Standardize all column names to lowercase for uniformity.
    4. Reorder the columns into a more logical order.
    5. Adjust the max allowed length of strings in outputs to easily see the full titles of movies
        - The movie title is treated as the priority for viewing string output in the dataframe.
    """)
    return


@app.cell
def _(Path, pl):
    _movie_metadata_path = Path.cwd() / "imdb_top_1000.csv"

    # Load the dataset.
    mm_raw = pl.read_csv(
        source=_movie_metadata_path, schema_overrides={"Released_Year": pl.String}
    )

    # Drop the columns mentioned above and rename columns as all lowercase.
    mm_transformed = (
        mm_raw
        .drop(["Poster_Link", "Overview", "Certificate"])
        .rename(lambda col_name: col_name.lower())
    )  # fmt: skip

    # Reorder columns.
    _col_order = [
        "released_year",
        "series_title",
        "director",
        "genre",
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

    mm_transformed = mm_transformed.select(_col_order)

    # Change the display so full titles are shown.
    _max_title_len = mm_transformed["series_title"].str.len_bytes().max()
    pl.Config.set_fmt_str_lengths(_max_title_len)

    mm_transformed
    return (mm_transformed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Identify Needed Transformations
    ### Columns Per Data Type
    """)
    return


@app.cell
def _(mm_transformed, pl):
    pl.Series(mm_transformed.dtypes).value_counts(sort=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### String Columns
    """)
    return


@app.cell
def _(cs, mm_transformed):
    mm_transformed.select(cs.string()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `released_year`, `runtime`, and `gross` should all be integer columns.

    #### Integer Columns
    """)
    return


@app.cell
def _(cs, mm_transformed):
    mm_transformed.select(cs.integer()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Float Columns
    """)
    return


@app.cell
def _(cs, mm_transformed):
    mm_transformed.select(cs.float()).columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `released_year` Values Preventing Casting to Integer Column
    """)
    return


@app.cell
def _(mm_transformed, pl):
    # Cast to integer by supressing the error that results. Then find the null values.
    mm_transformed.filter(pl.col("released_year").cast(pl.Int64, strict=False).is_null())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The only problem row preventing casting is the movie `"Apollo 13"`. The release year for this movie is `1995` and will be updated in the **Column Transformations** section below.

    ### Missing/Null Values
    """)
    return


@app.cell
def _(mm_transformed):
    mm_transformed.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Null values are only found in the `gross` and `meta_score` columns. This is not really an issue therefore no action will be taken for null values.

    ### Duplicates
    - Since each record should represent a unique film or TV series, we'll search for a candidate key by testing the following composite keys:
        - `(series_title,)`
        - `(series_title, director)`
        - `(series_title, director, released_year)`
    - Then we can assess what to do with any duplicates we find

    #### `(series_title,)`
    """)
    return


@app.cell
def _(mm_transformed, pl):
    mm_transformed.filter(pl.col("series_title").is_duplicated())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two movies that share the same title but are clearly different movies. No action is needed.

    #### `(series_title, director)`
    """)
    return


@app.cell
def _(mm_transformed, pl):
    # `pl.struct` is used to tie the two columns together into one object.
    mm_transformed.filter(pl.struct("series_title", "director").is_duplicated())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are no duplicates for the composite key `(series_title, director)`. Although not the goal of the project, if a database was constructed, this key could be used as primary key. Because no duplicates were found, it is unnecessary to test the third composite key.

    ## Transformations
    ### Column Transformations
    - `released_year` - There is one row value that is preventing the column from being cast as an integer column. The movie is `"Apollo 13"`; the `released_year` value is `"PG"` and should be changed to `"1995"`. Then the column can be cast to an unsigned 16-bit integer type. `pl.lit()` is needed because polars would otherwise try to look for a column named `"1995"`.
    - `genre` - The values need to be split on `", "`. The split operation will cast the column as a list type.
    - `runtime` - The characters `" min"` need to be removed. Then cast to an unsigned 16-bit integer type.
    - `gross` - All of the `","` characters need to be removed. Then cast to an unsigned 64-bit integer type.
    - `meta_score` - cast to an unsigned 8-bit integer type.
    - `no_of_votes` - cast to an unsigned 32-bit integer type.
    > Note: Casting to specific integer types is not strictly necessary here. It is mainly an exercise in evaluating the most minimal data type for each column that could be used in a possible database in a real industry project setting.
    """)
    return


@app.cell
def _(mm_transformed, pl):
    # Column Transformations.
    mm = (
        mm_transformed
        .with_columns(
            pl.when(pl.col("released_year") == "PG")
              .then(pl.lit("1995"))
              .otherwise(pl.col("released_year"))
              .cast(pl.UInt16)
              .alias("released_year"),
            pl.col("genre").str.split(", "),
            pl.col("runtime").str.strip_chars_end(characters=" min").cast(pl.UInt16),
            pl.col("gross").str.replace_all(pattern=",", value="", literal=True).cast(pl.UInt64),
            pl.col("meta_score").cast(pl.UInt8),
            pl.col("no_of_votes").cast(pl.UInt32),
        )
    )  # fmt: skip

    mm
    return (mm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Validate fix for Apollo 13 `release_year` Value
    """)
    return


@app.cell
def _(mm, pl):
    mm.filter(pl.col("series_title") == "Apollo 13")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Analysis
    ### Distributions of Numeric Columns
    """)
    return


@app.cell
def _(cs, go, make_subplots, mm, mo, product):
    # Select only the numeric columns.
    _numeric_data = mm.select(cs.numeric())

    # Subplot grid size.
    _subplot_rows = 2
    _subplot_cols = 3

    # Create a subplot figure.
    _fig = make_subplots(
        rows=_subplot_rows,
        cols=_subplot_cols,
        subplot_titles=_numeric_data.columns,
        horizontal_spacing=0.1,
        vertical_spacing=0.09,
    )

    # Subplot coordinates for placing the box plot for each column.
    _subplot_coords = product(range(1, _subplot_rows + 1), range(1, _subplot_cols + 1))

    for (_row, _col), _col_name in zip(_subplot_coords, _numeric_data.columns):
        _fig.add_trace(
            go.Box(y=_numeric_data[_col_name], name=_col_name), row=_row, col=_col
        )

    # Add size, title, and margins.
    _fig.update_layout(
        height=800,
        width=800,
        title={
            "text": "Numeric Column Distributions",
            "y": 0.965,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        margin=dict(t=85, l=30, r=30, b=30),
    )

    # Adjust the `gross` and `no_of_votes` to show a log scale.
    _fig.update_yaxes(type="log", row=1, col=3)
    _fig.update_yaxes(type="log", row=2, col=3)

    numeric_column_distributions_chart = mo.ui.plotly(_fig)
    numeric_column_distributions_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview of `mm`
    """)
    return


@app.cell
def _(mm):
    mm.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## NOTEBOOK IS CHECKED UP TO THIS CELL
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Handling Genres
    """)
    return


@app.cell
def _(mm, pl):
    mm.select(pl.col("genre").list.explode().unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are 21 unique genres. For analysis operations, the genres could be multi-hot encoded across 21 new columns. Instead, the functionality of the polars `List` type column will be used for the operations involving the genres.

    ## Analysis
    ### Top Directors Average IMDB Rating
    Find the 3 directors with the most movies. What is the average imdb score for each?
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.group_by("director")
        .agg(
            pl.len().alias("movie_count"),
            pl.mean("imdb_rating").round(2).alias("avg_imdb_rating"),
        )
        .sort(by="movie_count", descending=True)
        .head(3)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Actor Roles
    Find actors with the most leading roles (`star1`). Find the actors with the most roles.
    #### Actors with the most leading roles (`star1`)
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.group_by("star1")
        .len()
        .sort(by="len", descending=True)
        .head(10)
        .select(
            pl.col("star1").alias("actor"),
            pl.col("len").alias("leading_roles"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Actors with the most roles
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.select(
            pl.col("star1"),
            pl.col("star2"),
            pl.col("star3"),
            pl.col("star4"),
        )
        .unpivot()
        .group_by("value")
        .len()
        .sort(by="len", descending=True)
        .head(10)
        .select(
            pl.col("value").alias("actor"),
            pl.col("len").alias("roles"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Director/Actor Pairings
    For directors Steven Spielberg and Martin Scorsese, which actors have they worked with the most.
    #### Steven Spielberg
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.filter(pl.col("director") == "Steven Spielberg")
        .select(
            pl.col("star1"),
            pl.col("star2"),
            pl.col("star3"),
            pl.col("star4"),
        )
        .unpivot()
        .group_by("value")
        .len()
        .sort(by="len", descending=True)
        .head(3)
        .select(
            pl.col("value").alias("actor"),
            pl.col("len").alias("worked_with_spielberg"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Martin Scorsese
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.filter(pl.col("director") == "Martin Scorsese")
        .select(
            pl.col("star1"),
            pl.col("star2"),
            pl.col("star3"),
            pl.col("star4"),
        )
        .unpivot()
        .group_by("value")
        .len()
        .sort(by="len", descending=True)
        .head(3)
        .select(
            pl.col("value").alias("actor"),
            pl.col("len").alias("worked_with_scorsese"),
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Highest Rated Movie 2006-2016
    Find the highest rated movie in each year from 2006-2016.
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.filter(pl.col("released_year").is_between(2006, 2016))
        .group_by("released_year")
        .agg(
            pl.col("series_title")
            .filter(pl.col("imdb_rating") == pl.col("imdb_rating").max())
            .alias("highest_rated_movies"),
            pl.max("imdb_rating"),
        )
        .sort(by="released_year")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plots
    #### Relationship between `runtime` and `gross`
    """)
    return


@app.cell
def _(mm, pl, px):
    # Filter out movies that don't have a `gross` value.
    _filtered_mm = mm.filter(pl.col("gross").is_not_null())

    # Add 2 new columns representing the decade of release year and the imdb rating group.
    # \u2264 and \u2265 are less than or equal to and greater than or equal to respectively.
    _filtered_mm = _filtered_mm.with_columns(
        ((pl.col("released_year") // 10) * 10).alias("decade"),
        pl.col("imdb_rating")
        .cut(
            breaks=[8.0, 9.0],
            labels=["rating < 8", "8 \u2264 rating < 9", "rating \u2265 9"],
            left_closed=True,
        )
        .alias("imdb_rating_group"),
    ).sort(by="decade")

    _fig = px.scatter(
        _filtered_mm,
        x="runtime",
        y="gross",
        log_y=True,
        color="imdb_rating_group",
        facet_col="decade",
        facet_col_wrap=5,
        title="Gross Earnings vs. Runtime",
        labels={"runtime": "Runtime [min]", "gross": "Gross [$]", "decade": "Decade"},
        custom_data="series_title",
        opacity=0.6,
    )

    _fig.update_layout(
        height=400,
        width=1200,
        title={
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        margin=dict(t=65, l=30, r=30, b=30),
    )

    _fig.update_traces(
        hovertemplate="<b>Movie:</b> %{customdata}<br>"
        + "<b>Runtime:</b> %{x} minutes<br>"
        + "<b>Gross:</b> $%{y:,.0f}<extra></extra>",
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Relationship between `imdb_rating` and `gross`
    """)
    return


@app.cell
def _(mm, np, pl):
    min = 7.5
    max = 9.5
    # min = mm["imdb_rating"].min()
    # max = mm["imdb_rating"].max()
    width = 0.5
    n_bins = 5

    breaks = np.linspace(min, max, n_bins)

    a = (
        mm.select(
            pl.col("imdb_rating"),
            pl.col("imdb_rating")
            .cut(
                breaks=breaks,
            )
            .alias("cut"),
        )
        .sort(by="imdb_rating")
        .group_by("cut")
        .len()
        .sort(by="cut")
    )

    a
    return


@app.cell
def _(mm, pl, px):
    # Filter out movies that don't have a `gross` value.
    _filtered_mm = mm.filter(pl.col("gross").is_not_null())

    # Add 2 new columns representing the decade of release year and the imdb rating group.
    # _filtered_mm = (
    #     _filtered_mm
    #     .with_columns(
    #         ((pl.col("released_year") // 10) * 10).alias("decade"),
    #         pl.col("imdb_rating")
    #         .cut(
    #             breaks=[8.0, 9.0],
    #             labels=["rating < 8", "8 <= rating < 9", "rating >= 9"],
    #             left_closed=True,
    #         )
    #         .alias("imdb_rating_group"),
    #     )
    #     .sort(by="decade")
    # )

    # n_bins = 10
    # min_rating = mm["imdb_rating"].min()
    # max_rating = mm["imdb_rating"].max()

    _fig = px.histogram(
        mm,
        x="imdb_rating",
        y="gross",
        nbins=20,
        histfunc="avg",
        labels={"imdb_rating": "IMDB Rating", "gross": "Gross Earnings"},
        title="Movie Gross Earnings by IMDB Rating",
        hover_data=["series_title", "released_year"],
    )

    # fig = px.scatter(
    #     filtered_mm,
    #     x="imdb_rating",
    #     y="gross",
    #     log_y=True,
    #     color="decade",
    #     title="Gross Earnings vs. IMDB Rating",
    #     labels={"imdb_rating": "IMDB Rating", "gross": "Gross [$]"},
    #     hover_data=["series_title", "released_year"],
    #     color_continuous_scale="Plasma_r",
    # )

    _fig.update_layout(
        # height=400,
        # width=900,
        title={
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        # margin=dict(t=65, l=30, r=30, b=30),
        # coloraxis_colorbar={"title": "Decade"},
    )

    _fig.update_traces(
        hovertemplate="<b>Movie:</b> %{series_title}<br>"
        + "<b>Release Year:</b> %{customdata[1]}<br>"
        + "<b>IMDB Rating:</b> %{x}<br>"
        + "<b>Gross:</b> $%{y:,.0f}"
        + "<extra></extra>",
    )

    _fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Genres
    Average `imdb_Rating` rating per genre.
    """)
    return


@app.cell
def _(mm, pl):
    (
        mm.select(pl.col("genre"), pl.col("imdb_rating"))
        .explode("genre")
        .group_by("genre")
        .agg(
            pl.col("imdb_rating").mean().round(2).alias("avg_imdb_rating"),
            pl.len().alias("movie_count"),
        )
        .sort(by="movie_count", descending=True)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
