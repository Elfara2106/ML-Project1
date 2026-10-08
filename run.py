import numpy as np
import implementations
import helpers
import argparse 






def main():
    

    
    #True a modifier quand on run le test (juste pour le debug)
    x_train, x_test, y_train, train_ids, test_ids = helpers.load_csv_data("./data", True)
    #TODO faire le cross validation set 

    N, D = x_train.shape


    initial_w = np.zeros(D)

    #TODO implement terminal command to choose which method with arguments to choose
    #EXPLICATION: avec parser, on va faire passer en arguments le choix de la methode et ses parametres
    #L'idee est de lancer depuis le terminal sans modifier le code des quand on veut changer un parametre

    parser = argparse.ArgumentParser(description="Launch machine learning")

    #arguments
    parser.add_argument("--method", type=str, required=True, 
                        choices=['least_squares', 'ridge'], 
                        help="Method you want to used")
    parser.add_argument("--lambda_", type=float, default=0.1, 
                        help="regulariazation parameter")
    parser.add_argument("--max_iters", type=int, default=1000, 
                        help="Number of iteration")
    parser.add_argument("--gamma", type=float, default=1, 
                        help="parameter that define the learning rate")
    

    args = parser.parse_args()

    loss = np.inf



    if args.method == 'least_squares':
        w_train, loss = implementations.least_squares(y_train, x_train)
            
    elif args.method == 'ridge':
        w_train = implementations.ridge_regression(y_train, x_train, args.lambda_)

    elif args.method == 'mean_square_gd':
        loss, w_train = implementations.mean_squared_error_gd(y_train, x_train, args.gamma)

    elif args.method == 'mean_square_sgd':
            loss, w_train = implementations.mean_squared_error_sgd(y_train, x_train, initial_w, 1  ,args.max_iters, args.gamma)

    elif args.method == 'log':
        loss, w_train = implementations.logistic_regression(y_train, x_train, initial_w,args.max_iters, args.gamma)

    #reg logistic regression is missing

    print(loss)
    

    y_pred = x_test @ w_train
    print(y_pred)

    helpers.create_csv_submission(test_ids, y_pred, "result")

    
    
if __name__ == "__main__":
    
    main()