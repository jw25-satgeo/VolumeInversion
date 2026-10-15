"""Seasonal grouping over already-produced analysis records."""

SEASONS = {'growth': (10, 11, 12, 1, 2, 3, 4), 'melt': (5, 6, 7, 8, 9)}


def season_of(date, seasons=None):
    seasons = SEASONS if seasons is None else seasons
    month = int(str(date)[4:6])
    for name, months in seasons.items():
        if month in months:
            return name
    return None


def split_by_season(records, seasons=None):
    seasons = SEASONS if seasons is None else seasons
    out = {k: [] for k in seasons}
    for record in records:
        name = season_of(record['date'], seasons)
        if name is not None:
            out[name].append(record)
    return {k: v for k, v in out.items() if v}
