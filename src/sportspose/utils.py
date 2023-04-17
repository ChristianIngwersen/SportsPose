def chunks(lst, n, m=0):
    """Yield successive n-sized chunks from lst.
    Args:
        lst (list): List to be split into chunks.
        n (int): Group size
        m (int): Overlap size
    """
    for i in range(0, len(lst), n - m):
        yield lst[i : i + n]


SPORTSPOSE_CAMERA_INDEX_RIGHT = {
    "indoors": {
        "S00": 3,
        "S01": 3,
        "S02": 3,
        "S03": 3,
        "S04": 3,
        "S05": 3,
        "S06": 3,
        "S07": 3,
        "S08": 3,
        "S09": 3,
        "S10": 3,
        "S11": 3,
        "S12": 3,
        "S13": 3,
        "S14": 3,
        "S15": 2,
        "S16": 2,
        "S17": 2,
        "S18": 2,
        "S19": 2,
        "S20": 2,
        "S21": 2,
    },
    "outdoors": {
        "S00": 3,
        "S10": 3,
        "S22": 3,
        "S23": 3,
    },
}

VIEW_TO_SERIAL = {"right": "27087"}
