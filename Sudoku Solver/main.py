import cv2
import os
import numpy as np
import tensorflow as tf

import utils
from sudoku_solver_class import SudokuSolver


def solve_sudoku_puzzle(img_path, model_path):    
    # Check for valid filepath because cv2.imread fails silently
    if not os.path.exists(img_path):
        raise FileNotFoundError (f"File not found: '{img_path}'")
    img = cv2.imread(img_path)
    img = utils.resize_image(input_image=img, new_width=1000)

    # original image
    cv2.imshow("Original", cv2.resize(img, (int(img.shape[1] * 540 / img.shape[0]), 540)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load the trained model and make prediction
    loaded_model = tf.keras.models.load_model(model_path)

    # Locate grid cells in image
    cells, M, board_image = utils.extract_valid_cells(img)

    # Get the 2D array of the puzzle grid to be passed to the solver
    grid_array = utils.predict_grid(loaded_model, cells)

    # Create an instance of SudokuSolver and try to solve the puzzle
    solver = SudokuSolver(board=grid_array)
    solver.solve()

    # If there are no zeros left, the puzzle is solved. Display annotated image
    if not np.any(solver.board == 0):
        # Get the image of the board annotated with the solution
        print("Sudoku solved!")
        annotated_board_img = utils.render_solution(img, board_image, cells, solver.board, M)
        cv2.imshow("Answer", cv2.resize(annotated_board_img, (int(annotated_board_img.shape[1] * 540 / annotated_board_img.shape[0]), 540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Could not solve the puzzle. Check for misclassified digits.\n")

    solver.print_board()


if __name__ == "__main__":
    IMG = "sudoku/2.png" # add path of image
    MODEL = "model.h5" # add path of model

    solve_sudoku_puzzle(IMG, MODEL)