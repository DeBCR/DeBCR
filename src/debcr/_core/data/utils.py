import os

def get_format(file_path: str) -> str:
    """Returns the file format for the give file path."""
    _,file_format = os.path.splitext(file_path)
    return file_format[1:] # truncate preceding dot

'''
def is_supported_format(file_format):
    """Returns True if the loader for the given format is implemented."""
    file_format = file_format.lower()
    is_supported = file_format in FORMAT_REGISTRY
    return is_supported
'''