########################################################################
########################## IMPORTAR LIBRERÍAS ##########################
########################################################################

# Manejo de parámetros
import os
import sys

# Módulos
import eye_report_utils as utils


########################################################################
############################ Command-Line ##############################
########################################################################

if __name__ == "__main__":

############################ DEFINIR PATHS #############################

    # Obtenemos el path completo a partir del input por terminal
    input_path = utils.path_input(sys.argv[1])

    # directorio de salida
    if os.path.exists(sys.argv[1]): # si ya es ruta completa
        output_path = sys.argv[1]
    else:
        # si es ruta parcial
        output_path = os.path.join(os.getcwd(), sys.argv[1])

############################## EJECUCIÓN ###############################

    utils.eye_report(input_path, output_path=output_path, plot=False)