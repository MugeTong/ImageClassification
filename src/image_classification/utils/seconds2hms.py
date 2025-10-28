def seconds2hms(seconds: float) -> str:
    """Converts seconds to a string in the format 'HH:MM:SS'.

    Args:
        seconds (float): The total number of seconds to convert.

    Returns:
        str: A string representing the time in 'HH:MM:SS' format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"
