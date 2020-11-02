import os
import time
import datetime

user_action = {}
model = []
#9大状态
states = ["SL", "SE", "SG", "ML", "ME", "MG", "LL", "LE", "LG"]

validate_path = os.path.join(os.getcwd(), "validate.txt")
model_path = os.path.join(os.getcwd(), "model.txt")

#读取validate data
with open(validate_path, 'r', encoding='utf-8') as f_read:
    for line in f_read:
        items = line.strip().split(',')
        user_id = items[0]
        if user_id in user_action.keys():
            hist = user_action[user_id]
            lst = [items[2], items[-1]]
            hist.append(lst)
        else:
            hist = []
            hist.append([items[2], items[-1]])
            user_action[user_id] = hist
print(user_action)

#读取model data
with open(model_path, 'r', encoding='utf-8') as f_read:
    for line in f_read:
        items = line.strip().split()
        row = []
        for item in items:
            row.append(float(item))
        model.append(row)
print(model)

#根据最近客户的行为数据(至少两次交易)make prediciton
for user_id,user_action_list in user_action.items():
    if len(user_action_list) < 2:
        continue
    state_sequence = []
    last_date = ''
    prior = user_action_list[0]
    for i in range(1, len(user_action_list)):
        current = user_action_list[i]
        prior_date = prior[0]
        current_date = current[0]

        #相隔天数
        prior_date = time.strptime(prior_date, '%Y-%m-%d')
        current_date = time.strptime(current_date, '%Y-%m-%d')
        prior_date = datetime.datetime(prior_date[0], prior_date[1], prior_date[2])
        current_date = datetime.datetime(current_date[0], current_date[1], current_date[2])
        days_diff = (current_date - prior_date).days

        dd = 'L'
        if days_diff < 30:
            dd = 'S'
        elif days_diff < 60:
            dd = 'M'

        #相差金额
        prior_amount = int(prior[1])
        current_amount = int(current[1])

        ad = 'G'
        if prior_amount < 0.9 * current_amount:
            ad = 'L'
        elif prior_amount < 1.1 * current_amount:
            ad = 'E'

        state_sequence.append(dd+ad)

        prior = current
        last_date = current_date

    if state_sequence:
        #根据最近一个状态发送营销邮件日期
        last_state = state_sequence[-1]
        row_index = states.index(last_state)
        row_value = model[row_index] #转移矩阵中行号为row_index的这一行值
        max_value = max(row_value) #row_value中最大值
        col_index = row_value.index(max_value) #max_value的索引号
        next_state = states[col_index]

        if next_state.startswith('S'):
            next_date = last_date + datetime.timedelta(15)
        elif next_state.startswith('E'):
            next_date = last_date + datetime.timedelta(45)
        else:
            next_date = last_date + datetime.timedelta(90)

    print('用户：{},下一次营销邮件：{}'.format(user_id, next_date))
