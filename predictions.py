import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

pd.set_option('display.max_columns', None)


# Part 1: Loading, Cleaning and Setting Data Up
def process_data(train_list_years, test_list_years):
    """
    Takes input of desired years to use in training dataset and testing dataset.
    Outputs testing and training dataframes to be further reshaped.
    Final stage accounts for processing team needs data into a dictionary.
    """

    # Load Data
    prospects = pd.read_csv('nfl_draft_prospects.csv')
    profiles = pd.read_csv('nfl_draft_profiles.csv')
    order = pd.read_csv(f'draft_order_{str(test_list_years[0])}.csv')
    needs = pd.read_csv(f'team_needs_{str(test_list_years[0])}.csv')

    extra_cols_pr = ["player_image", "position", 
                "school_name", "school_abbr", "link", 
                "team", "team_logo_espn", "guid", "height", 
                "weight", "traded", "trade_note", "team_abbr"]
    prospects = prospects.drop(extra_cols_pr, axis=1)

    extra_cols_pf = ["guid", "alt_player_id", "position", 
                     "weight", "height", "player_image", "link", 
                     "school_logo", "school_name", "school_abbr", 
                     "pos_rk", "ovr_rk", "grade", "school", "pos_abbr"]
    profiles = profiles.drop(extra_cols_pf, axis=1)
    
    # Handle Position Column and Numerical Encoding
    master_df = pd.merge(prospects, profiles, on="player_id")
    """
    Mapping Key for Positions:
    0: QB, 1: DE, 2: OT, 3: CB, 4: ILB,
    5: S, 6: OLB, 7: DT, 8: RB, 9: IOL,
    10: TE, 11: WR, 12: FB, 13: P, 14: PK
    """
    pos_abbr_encoding = {
        0: "QB", 1: "DE", 2: "OT", 3: "CB", 
        4: "ILB", 5: "S", 6: "OLB", 7: "DT", 
        8: "RB", 9: "IOL", 10: "TE", 11: "WR", 
        12: "FB", 13: "P", 14: "PK"
    }

    # Converts obscure pos_abbr to more common ones
    for i, abbr in enumerate(master_df["pos_abbr"]):
        if abbr == "OG" or abbr == "C" or abbr == "LS":
            master_df.at[i, "pos_abbr"] = 9
            continue;

        if abbr == "DB":
            master_df.at[i, "pos_abbr"] = 3
            continue;

        if abbr == "LB":
            master_df.at[i, "pos_abbr"] = 4
            continue;

        for key, value in pos_abbr_encoding.items():
            if value == abbr:
                master_df.at[i, "pos_abbr"] = int(key)        

        if master_df.loc[i, "pos_abbr"] not in pos_abbr_encoding.keys():
            print(master_df.loc[i, "player_name_x"], master_df.loc[i, "draft_year"], master_df.loc[i, "pos_abbr"])

    # Redone pos_rk column based on new positions
    for year in range(2011, 2021):
        pos_ranks = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 
                7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1}
        for index, row in master_df[master_df["draft_year"] == year].iterrows():
            if row["pos_abbr"] not in pos_ranks.keys():
                print(row["player_name_x"], row["draft_year"], row["pos_abbr"])
            new_pos_rk = pos_ranks.get(row["pos_abbr"])
            master_df.at[index, "pos_rk"] = int(new_pos_rk)
            pos_ranks[row["pos_abbr"]] += 1

    for index, row in master_df.iterrows():
        if pd.isna(row["pos_rk"]):
            print(row["player_name_x"], row["draft_year"], row["pos_abbr"])

    # Impute missing grades with average value
    master_df["grade"].fillna(master_df["grade"].mean(), inplace=True)    

    # Impute missing ovr_rk with overall column
    for index, row in master_df.iterrows():
        if pd.isna(row["ovr_rk"]) and not pd.isna(row["overall"]):
            master_df.at[index, "ovr_rk"] = row["overall"]
    
    # Training Data (as inputted, default = 2017-2019)
    master_df_train = master_df[master_df["draft_year"].isin(train_list_years)]

    # Testing Data (as inputted, default = 2020)
    master_df_test = master_df[master_df["draft_year"].isin(test_list_years)]

    # Convert order csv to a dictionary
    order_dict = order.set_index('pick')['team_abbr'].to_dict()

    # Convert needs csv to a dictionary
    needs_dict = needs.set_index('team_abbr')[['need1', 'need2', 'need3', 'need4', 'need5']].apply(list, axis=1).to_dict()
    
    return master_df_train, master_df_test, order_dict, needs_dict


# Part 2: Training the Model
def train_model(training_df, model):
    """
    Takes input of dataframes from process_data.
    Version A1: RF trained with pos_abbr, grade, pos_rk, ovr_rk from 2017-2019. Tested on 2020.
    Version A2: Test out different values of RF's n_estimators. Trying out other regression models.
    Version A3: Trains with same factors but from 2013-2019. Tested on 2020-2021.
    
    Version B1: Includes NLP for player comments.
    Version B2: Substituting basic neural network for RF in B1.
    Version B3: Fine tuning with neural network parameters. 

    Version C1: Develop working draft simulator using team need.

    Version D1: Final feature engineering, using weight, height, and more.
    Outputs a trained model on dataset given.
    """

    x_train = []
    y_train = []
    
    # Constructing x_train array with pos_abbr, grade, pos_rk, ovr_rk and y_train array with overall
    for index, row in training_df.iterrows():
        position = row["pos_abbr"]
        grade = float(row["grade"])
        pos_rk = int(row["pos_rk"])
        ovr_rk = int(row["ovr_rk"])

        x_train.append([position, grade, pos_rk, ovr_rk])

        if not pd.isna(row["overall"]):
            y_train.append(int(row["overall"]))
        else:
            y_train.append(int(training_df.loc[index, "ovr_rk"]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = model.fit(x_train, y_train)

    return model


# Part 3: Making Predictions with the Model
def model_predict(testing_df, model):
    """
    Constructs testing arrays and makes predictions. 
    Returns both the predicted and true arrays and a list of names to keep track. 
    """
    x_test = []
    y_preds = []
    y_real = []
    names_in_order = []
    pos_in_order = []

    # Constructing x_test array with same columns array
    for index, row in testing_df.iterrows():
        position = row["pos_abbr"]
        grade = float(row["grade"])
        pos_rk = int(row["pos_rk"])
        ovr_rk = int(row["ovr_rk"])

        x_test.append([position, grade, pos_rk, ovr_rk])

        names_in_order.append(row["player_name_x"])
        pos_in_order.append(row["pos_abbr"])

    x_test = np.array(x_test)

    # Making predictions from x_test
    y_preds = model.predict(x_test)
    y_preds = np.array(y_preds)


    # Constructing actual y_test
    for index, row in testing_df.iterrows():
        if not pd.isna(row["overall"]):
            y_real.append(int(row["overall"]))
        else:
            y_real.append(int(testing_df.loc[index, "ovr_rk"]))

    y_real = np.array(y_real)

    return y_preds, y_real, names_in_order, pos_in_order


# Part 4: Evaluating the Predictions 
def eval(preds, true, names):
    """
    Calculates some statistical metrics to evaluate the model, prints a few of the predictions.
    Returns 2 of the metrics (rmspe and r-value).
    """

    # Calculating metrics
    idx = np.where(true != 0)
    rmspe = (np.sqrt(np.mean(np.square((true[idx] - preds[idx]) / true[idx]))))

    pearson = np.corrcoef(preds, true)[0][1]

    residuals = []
    for i in range(preds.shape[0]):
        residuals.append([names[i], preds[i], true[i]])

    return rmspe, pearson, residuals #, matches


# Metrics to comprehensively evaluate accuracy of simulation models
def final_eval(simmed, true, k=3):
    mae = np.mean(np.abs(np.array(simmed) - np.array(true)))

    matches = 0
    w = 0
    for sim, act in zip(simmed, true):
        distance = abs(sim - act)
        if distance < k:
            w = 1
        else:
            if 1 - (distance / act) > 0:
                w = 1 - (distance / act)
        matches += w

    top_k = matches / len(simmed) * 100

    return mae, top_k


# Fully Working Draft Simulator w/BPA strategy
def bpa(preds, true, names, order_dict):
    """
    Similar to eval() function; operates one step further by assigning players to teams. 
    """
    preds_list = list(preds.copy())
    true_list = list(true.copy())
    names_list = list(names.copy())
    matches = 0

    simmed = []
    actuals = []
    
    for i in range(32):
        team = order_dict.get(i+1)
        # Look for player with lowest predicted pick (BPA strategy)
        min_index = preds_list.index(min(preds_list))
        player = names_list[min_index]
        
        actual = true_list[min_index]

        names_list.pop(min_index)
        true_list.pop(min_index)
        preds_list.pop(min_index)
        
        print(f"Round 1, Pick {i+1}: {team} selects {player}")
        true_player = names[i]
        if player == true_player:
            matches += 1
        
        simmed.append(i+1)
        actuals.append(actual)

    mae, tk = final_eval(simmed, actuals)

    matches /= 32
    return matches, mae, tk


# Helper to calculate how much a team needs a specific position
def need_weight(team, pos_index, needs_dict, pos_abbr_encoding):
    """
    Given the team abbr., index # of a position, the mapping from teams to their needs,
    and the position/abbreviation encoding, the function returns the weight of that need
    for the given team.
    """

    position = pos_abbr_encoding[pos_index] 
    if team not in needs_dict:
        return 0  
    
    team_needs = needs_dict[team]          
    if position in team_needs:
        idx = team_needs.index(position)     
        return 5 - idx       
             
    else:
        return 0


# Fully Working Draft Simulator based on Team Needs
def final_sim(preds, true, names, positions, order_dict, needs_dict, n=4): 
    """
    Final draft simulator; builds on bpa by accounting for team needs.
    preds: predicted pick via ML of the player
    true: actual position selected of the player
    names: name of the player
    positions: position index of the player
    order_dict: team: pick number, for all picks in draft
    needs_dict: team: 5 most needed positions, for all picks in draft
    n: weighting of need vs. player talent
    """
    pos_abbr_encoding = {
        0: "QB", 1: "DE", 2: "OT", 3: "CB", 
        4: "ILB", 5: "S", 6: "OLB", 7: "DT", 
        8: "RB", 9: "IOL", 10: "TE", 11: "WR", 
        12: "FB", 13: "P", 14: "PK"
    }
    
    simmed = []
    actuals = []

    # Sort preds, players, names accordingly
    draftboard = []

    for i in range(len(preds)):
        draftboard.append((preds[i], true[i], names[i], positions[i])) 

    draftboard.sort(key=lambda t: t[0])
    
    # For each team, iterative process to assign a player
    matches = 0

    for pick_number in range(1, 33):
        team = order_dict.get(pick_number, "??") 
        
        # Find the player with the best "score" for this team
        best_score = float('-inf')
        best_idx = -1
        chosen_player = None

        for idx, (pred_pick, actual_pick, p_name, p_index) in enumerate(draftboard):
            score = n * (need_weight(team, p_index, needs_dict, pos_abbr_encoding)) - pred_pick

            if score > best_score:
                best_score = score
                chosen_player = (pred_pick, actual_pick, p_name, p_index)
                best_idx = idx

        # "Draft" the chosen player and remove them from the board
        if chosen_player:
            draftboard.pop(best_idx)

            pred_pick, actual_pick, player_name, p_index = chosen_player
            
            # Print which player was selected
            pos_str = pos_abbr_encoding[p_index]
            print(f"Round 1, Pick {pick_number}: {team} selects {player_name} "
                  f"[pred_pick={pred_pick:.1f}, actual={actual_pick}]")

            # Count a "match" if this player's 'true' pick == pick_number
            if actual_pick == pick_number:
                matches += 1

        simmed.append(pick_number)
        actuals.append(actual_pick)
    
    mae, tk = final_eval(simmed, actuals)

    matches /= 32
    return matches, mae, tk


if __name__ == "__main__":
    train_df, test_df, order_dict, needs_dict = process_data([2015, 2016, 2018, 2019], [2017])
    model = RandomForestRegressor(n_estimators = 100, random_state = 42)
    
    #model = DecisionTreeRegressor()
    #model = SVR(kernel='linear')
    trained_model = train_model(train_df, model)
    preds, true, names, positions = model_predict(test_df, trained_model)

    # Testing initial model
    """
    stats = eval(preds, true, names)
    #print(f"Match %: {round(stats[3], 5) * 100}%")
    
    print(f"Root Mean Squared Percentage Error: {stats[0]}, Correlation Coefficient: {stats[1]}")
    
    print("Residuals:")

    for i in range(10):
        print(f"Name: {stats[2][i][0]}, Predicted Pick: {stats[2][i][1]}, Real Pick: {stats[2][i][2]}")
    """

    
    # Testing BPA model
    #matches_bpa, mae_bpa, tk_bpa = bpa(preds, true, names, order_dict)
    #print(f"Match %: {round(matches_bpa, 5) * 100}%")
    #print(f"MAE: {mae_bpa}, Top-K: {tk_bpa}")
    
    # Testing final model
    matches_final, mae_final, tk_final = final_sim(preds, true, names, positions, order_dict, needs_dict)
    print(f"Match %: {round(matches_final, 5) * 100}%")
    print(f"MAE: {mae_final}, Top-K: {tk_final}")

    