import sys

def error_msg_detail(errors, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_ms = "Error Occured in script anme [{0}] at line number [{1}] error message is [{2}]" .format(file_name, exc_tb.tb_lineno, str(errors))


