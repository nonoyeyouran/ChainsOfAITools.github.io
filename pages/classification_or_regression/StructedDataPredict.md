# 这是一个对于结构化数据进行机器学习的工具，针对没有编程基础和机器学习基础的应用者

---
## 目录
---
- [结构化数据是什么](#结构化数据是什么)
- [环境配置](#环境配置)
- [代码](#代码)
- [使用示例](#使用示例)

---

## 结构化数据是什么

---

## 环境配置
需要配置python环境，安装必要的python包

---

## 代码
```
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import joblib
import argparse

# 1. 处理数据
def handleData(file_path, feas_num, standard_scaler_path):
    # 1.1 读取数据
    file_input = open(file_path, encoding="utf8")
    data_temp = []
    for line in file_input:
        if line.startswith('\ufeff'):
            temp_list = line[1:].strip().split(",")
            data_temp.append(temp_list)
        else:
            temp_list = line.strip().split(",")
            data_temp.append(temp_list)
    data_origin = np.array(data_temp, dtype=np.float32)
    data_X = data_origin[:, 0:feas_num]
    data_Y = data_origin[:, feas_num]
    #print(data_X)
    #print(data_Y)
    # 1.2 处理数据
    # 标准化
    scaler = StandardScaler()
    data_X_stand = scaler.fit_transform(data_X)
    # 高阶特征
    poly = PolynomialFeatures(degree=2)
    data_X_poly = poly.fit_transform(data_X_stand)
    #print(poly.get_feature_names_out())
    joblib.dump(scaler, standard_scaler_path)
    return (data_X_stand, data_X_poly, data_Y, poly.get_feature_names_out())

def handlePredictData(file_path, feas_num, standard_scaler_path):
    # 1.1 读取数据
    file_input = open(file_path, encoding="utf8")
    data_temp = []
    for line in file_input:
        if line.startswith('\ufeff'):
            temp_list = line[1:].strip().split(",")
            data_temp.append(temp_list)
        else:
            temp_list = line.strip().split(",")
            data_temp.append(temp_list)
    data_origin = np.array(data_temp, dtype=np.float32)
    data_X = data_origin[:, 0:feas_num]
    # 1.2 处理数据
    # 标准化
    scaler = joblib.load(standard_scaler_path)
    data_X_stand = scaler.fit_transform(data_X)
    # 高阶特征
    poly = PolynomialFeatures(degree=2)
    data_X_poly = poly.fit_transform(data_X_stand)
    #print(poly.get_feature_names_out())
    return (data_X_stand, data_X_poly, poly.get_feature_names_out())

def get_model_and_params(model_type, param_grid):
    is_params_grid_empty = True
    if len(param_grid.keys()) > 0:
        is_params_grid_empty = False
    if model_type == "Line":
        model = LinearRegression()
        param_grid = {}
    elif model_type == "Ridge":
        model = Ridge(alpha=75.0)
        if not is_params_grid_empty:
            model = Ridge(alpha=param_grid["alpha"])
        else:
            param_grid = {
                'alpha': np.arange(1.0, 100.0, 1.0),
            }
    elif model_type == "Lasso":
        model = Lasso(alpha=0.0001)
        if not is_params_grid_empty:
            model = Lasso(alpha=param_grid["alpha"])
        else:
            param_grid = {
                'alpha': np.arange(0.0001, 0.1, 0.0001),
            }
    elif model_type == "XGB":
        model = XGBRegressor(random_state=42)
        if not is_params_grid_empty:
            model = XGBRegressor(n_estimators=param_grid["n_estimators"], learning_rate=param_grid["learning_rate"], max_depth=param_grid["max_depth"], random_state=42)
        else:
            param_grid = {
                'n_estimators': np.arange(50, 100, 10),
                'learning_rate': np.arange(0.01, 0.1, 0.01),
                'max_depth': np.arange(3, 8, 1)
            }
    elif model_type == "RF":
        model = RandomForestRegressor(random_state=42)
        if not is_params_grid_empty:
            model = RandomForestRegressor(n_estimators=param_grid["n_estimators"], min_samples_split=param_grid["min_samples_split"], max_depth=param_grid["max_depth"], random_state=42)
        else:
            param_grid = {
                'n_estimators': np.arange(50, 100, 10),
                'max_depth': np.arange(3, 8, 1),
                'min_samples_split': np.arange(2, 10, 2)
            }
    elif model_type == "SVR":
        model = SVR(kernel='rbf')
        if not is_params_grid_empty:
            model = SVR(kernel='rbf', C=param_grid["C"], epsilon=param_grid["epsilon"])
        else:
            param_grid = {
                'C': np.arange(0.01, 0.2, 0.01),
                'epsilon': np.arange(0.006, 0.008, 0.0005)
            }
    return (model, param_grid)

def K_fold(X, Y, model):
    rmse_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        # 分割训练集和测试集
        # print(train_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        # 训练模型
        model.fit(X_train, y_train)
        # print("模型系数:", model.coef_)
        # 预测并计算均方误差
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)
    return np.mean(rmse_scores)

def select_feas_using_target_model(model_type, selected_feas, X, Y):
    final_selected_feas = []
    param_grid = {}
    model, param_grid = get_model_and_params(model_type, param_grid)
    X_selected = X[:, selected_feas]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_selected, Y)
    model, _ = get_model_and_params(model_type, grid.best_params_)
    all_rmse = -grid.best_score_
    for feas in selected_feas:
        temp_feas = selected_feas.copy()
        temp_feas.remove(feas)
        X_selected = X[:, temp_feas]
        model, _ = get_model_and_params(model_type, grid.best_params_)
        if K_fold(X_selected, Y, model) > all_rmse:
            final_selected_feas.append(feas)
    if len(final_selected_feas) == 0:
        final_selected_feas = selected_feas
    return final_selected_feas

def feas_selection(inputs, feas_type):
    X, X_poly, Y, poly_feas_names = inputs
    X_ = X
    if feas_type == 2:
        X_ = X_poly
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = Lasso(alpha=0.0006)
    alpha_list = np.arange(0.0001, 0.1, 0.0001)
    param_grid = {
        'alpha': alpha_list
    }
    grid = GridSearchCV(model, param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_, Y)
    print("Lasso最佳参数:", grid.best_params_)
    print("Lasso最佳平均RMSE:", -grid.best_score_)
    model = Lasso(alpha=grid.best_params_["alpha"])
    model.fit(X_, Y)
    selected_feas = []
    for i, item in enumerate(model.coef_):
        if abs(item) > 0.0000000001:
            selected_feas.append(i)
    selected_feas_by_model = select_feas_using_target_model("Lasso", selected_feas, X_, Y)
    return (selected_feas, selected_feas_by_model)

def trainModel(inputs, feas_type, model_type, is_grid_search, is_feature_selection, selected_feas, selected_feas_by_lasso, feature_selection_type):
    print("************************************************")
    X, X_poly, Y, poly_feas_names = inputs
    X_ = X
    if feas_type == 2:
        X_ = X_poly
    param_grid = {}
    model, param_grid = get_model_and_params(model_type, param_grid)

    final_selected_feas = np.arange(0, len(X_[0]), 1)
    if is_feature_selection:
        if feature_selection_type == "R1":
            X_ = X_[:, selected_feas]
            final_selected_feas = selected_feas
        elif feature_selection_type == "Lasso":
            X_ = X_[:, selected_feas_by_lasso]
            final_selected_feas = selected_feas_by_lasso
        elif feature_selection_type == "Model":
            final_selected_feas = select_feas_using_target_model(model_type, selected_feas, X_, Y)
            X_ = X_[:, final_selected_feas]
    print(model_type + " " + feature_selection_type + " selected feas:")
    print(final_selected_feas)

    if is_grid_search:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_, Y)
        print(model_type, "最佳参数:", grid.best_params_)
        print(model_type, "最佳平均RMSE:", -grid.best_score_)
        return (grid.best_params_, -grid.best_score_, final_selected_feas)
    else:
        score = K_fold(X_, Y, model)
        print(model_type, "平均RMSE:", score)
        return (None, score, final_selected_feas)

def trainWithAllData(X, Y, model):
    model.fit(X, Y)


def main(args):
    # 处理传入的参数
    op_type = args.op_type
    model_path = args.model_path
    scaler_path = args.scaler_path
    config_path = args.config_path
    feas_num = args.feas_num
    use_cross_feas = args.use_cross_feas
    use_feas_selection = args.use_feas_selection
    if op_type == "train":
        train_path = args.train_path
        data_handled = handleData(train_path, feas_num=feas_num, standard_scaler_path=scaler_path)

        feas_type = 1
        grid_search = True
        is_feature_selection = False
        selected_feas = None
        model_type_list = ["Line", "Ridge", "Lasso", "XGB", "RF", "SVR"]
        best_params = None
        best_score = 1000
        best_model_type = ""
        best_feas_type = 1
        best_final_selected_feas = []
        best_is_feature_selection = is_feature_selection
        print("腐蚀速率学习-一阶特征：")
        for model_type in model_type_list:
            params, score, final_selected_feas = trainModel(data_handled, feas_type, model_type=model_type, is_grid_search=grid_search, is_feature_selection=is_feature_selection, selected_feas=[], selected_feas_by_lasso=[], feature_selection_type="None")
            if score < best_score:
                best_params = params
                best_score = score
                best_model_type = model_type
                best_feas_type = feas_type
                best_is_feature_selection = is_feature_selection

        if use_cross_feas == 1:
            feas_type = 2
            print("腐蚀速率学习-二阶特征：")
            for model_type in model_type_list:
                params, score, final_selected_feas = trainModel(data_handled, feas_type, model_type=model_type, is_grid_search=grid_search, is_feature_selection=is_feature_selection, selected_feas=[], selected_feas_by_lasso=[], feature_selection_type="None")
                if score < best_score:
                    best_params = params
                    best_score = score
                    best_model_type = model_type
                    best_feas_type = feas_type
                    best_is_feature_selection = is_feature_selection
            if use_feas_selection == 1:
                is_feature_selection = True
                selected_feas, selected_feas_by_model = feas_selection(data_handled, feas_type)
                print("腐蚀速率学习-二阶特征+特征选择：")
                for model_type in model_type_list:
                    params, score, final_selected_feas = trainModel(data_handled, feas_type, model_type=model_type,
                                                                    is_grid_search=grid_search,
                                                                    is_feature_selection=is_feature_selection,
                                                                    selected_feas=selected_feas,
                                                                    selected_feas_by_lasso=selected_feas_by_model,
                                                                    feature_selection_type="R1")
                    if score < best_score:
                        best_params = params
                        best_score = score
                        best_model_type = model_type
                        best_feas_type = feas_type
                        best_is_feature_selection = is_feature_selection
                        best_final_selected_feas = final_selected_feas
                    params, score, final_selected_feas = trainModel(data_handled, feas_type, model_type=model_type,
                                                                    is_grid_search=grid_search,
                                                                    is_feature_selection=is_feature_selection,
                                                                    selected_feas=selected_feas,
                                                                    selected_feas_by_lasso=selected_feas_by_model,
                                                                    feature_selection_type="Lasso")
                    if score < best_score:
                        best_params = params
                        best_score = score
                        best_model_type = model_type
                        best_feas_type = feas_type
                        best_is_feature_selection = is_feature_selection
                        best_final_selected_feas = final_selected_feas
                    params, score, final_selected_feas = trainModel(data_handled, feas_type, model_type=model_type,
                                                                    is_grid_search=grid_search,
                                                                    is_feature_selection=is_feature_selection,
                                                                    selected_feas=selected_feas,
                                                                    selected_feas_by_lasso=selected_feas_by_model,
                                                                    feature_selection_type="Model")
                    if score < best_score:
                        best_params = params
                        best_score = score
                        best_model_type = model_type
                        best_feas_type = feas_type
                        best_is_feature_selection = is_feature_selection
                        best_final_selected_feas = final_selected_feas

        print("best_model_type:" + best_model_type)
        print("best RMSE:" + str(best_score))
        final_model, _ = get_model_and_params(model_type=best_model_type, param_grid=best_params)
        file_config = open(config_path, 'w', encoding="utf8")
        if best_feas_type == 1:
            trainWithAllData(data_handled[0], data_handled[2], final_model)
            file_config.writelines("feas_type:" + str(best_feas_type) + "\n")
            file_config.writelines("final_selected_feas:" + ",".join([str(i) for i in best_final_selected_feas]) + "\n")
        elif best_feas_type == 2:
            if best_is_feature_selection:
                trainWithAllData(data_handled[1][:, best_final_selected_feas], data_handled[2], final_model)
                file_config.writelines("feas_type:" + str(best_feas_type) + "\n")
                file_config.writelines(
                    "final_selected_feas:" + ",".join([str(i) for i in best_final_selected_feas]) + "\n")
            else:
                trainWithAllData(data_handled[1], data_handled[2], final_model)
                file_config.writelines("feas_type:" + str(best_feas_type) + "\n")
                file_config.writelines(
                    "final_selected_feas:" + ",".join([str(i) for i in best_final_selected_feas]) + "\n")
        file_config.close()
        joblib.dump(final_model, model_path)
    elif op_type == "predict":
        predict_path = args.predict_path
        predict_X, predict_poly_X, _ = handlePredictData(predict_path, feas_num=feas_num, standard_scaler_path=scaler_path)
        load_model = joblib.load(model_path)
        file_config = open(config_path, encoding="utf8")
        configs = {}
        for line in file_config:
            if len(line.strip()) > 0:
                key, val = line.strip().split(":")
                if key == "final_selected_feas":
                    if len(val) > 0:
                        configs[key] = [int(i) for i in val.split(",")]
                else:
                    configs[key] = val
        if configs["feas_type"] == "1":
            print(load_model.predict(predict_X))
        elif configs["feas_type"] == "2":
            if "final_selected_feas" in configs.keys():
                print(load_model.predict(predict_poly_X[:, configs["final_selected_feas"]]))
            else:
                print(load_model.predict(predict_poly_X))

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="一个简单的示例程序")

    # 添加参数
    parser.add_argument('--op_type', type=str, required=True, help='操作类型，训练或预测')
    parser.add_argument('--feas_num', type=int, required=True, help='特征数')
    parser.add_argument('--train_path', type=str, required=False, help='训练数据地址')
    parser.add_argument('--predict_path', type=str, required=False, help='预测数据地址')
    parser.add_argument('--model_path', type=str, required=False, help='模型保存地址')
    parser.add_argument('--scaler_path', type=str, required=False, help='标准化器保存地址')
    parser.add_argument('--config_path', type=str, required=False, help='模型参数配置保存地址')
    parser.add_argument('--use_cross_feas', type=int, required=False, help='是否使用二阶交叉特征')
    parser.add_argument('--use_feas_selection', type=int, required=False, help='是否使用特征选择')

    # 解析参数
    args = parser.parse_args()

    # 调用 main 函数并传入参数
    main(args)
```
