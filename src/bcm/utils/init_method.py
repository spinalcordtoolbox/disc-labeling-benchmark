import numpy as np

def init_method(txt_lines, method_name):
    """
    Add method to txt_lines

    :param txt_lines: list of lists containing all the txt file content
    :param method_name: method that have to be added to the text lines

    :return: txt_lines with the method initialized
    """
    # Convert list of lists to numpy array
    txt_lines_np = np.array(txt_lines)

    # Extract number of lines
    num_line = txt_lines_np.shape[0]
    
    # Extract number of columns
    num_col = txt_lines_np.shape[1]

    # Remove \n from last column
    last_col = np.transpose([[line[-1].replace('\n','') for line in txt_lines]])

    # Initialize new method column
    new_col = [method_name + '\n'] + ['None\n']*(num_line-1)
    new_col = np.transpose([new_col])

    # Merge all the columns
    new_txt_lines_np = np.concatenate((txt_lines_np[:,:num_col-1], last_col, new_col), axis=1)

    return new_txt_lines_np.tolist()