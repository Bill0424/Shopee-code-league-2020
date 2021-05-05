import time
from datetime import datetime, timedelta
import csv

def find_location(data):
    if data.lower().find('metro manila') != -1:
        return 0
    elif data.lower().find('luzon') != -1:
        return 1
    elif data.lower().find('visayas') != -1:
        return 3
    else:
        return 3

def find_sla(start, end):
    print(start, end)
    if start+end == 0:
        return 3
    elif start+end == 1 or start+end == 2:
        return 5
    else:
        return 7

def how_many_day(start, end):
    how_day = 0
    while start != end:
        start += timedelta(days=1)
        if pick.weekday() != 6 and pick not in holiday:
            how_day += 1
    return how_day

holiday = [(datetime.strptime('2020-3-25', "%Y-%m-%d")), (datetime.strptime('2020-3-30', "%Y-%m-%d")), (datetime.strptime('2020-3-31', "%Y-%m-%d"))]

with open('final.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['orderid', 'is_late'])
    with open("/Users/bill/Downloads/logistics-shopee-code-league/delivery_orders_march.csv", newline='') as data_set:
        rows = csv.reader(data_set, skipinitialspace=True)
        a = 0
        for data in rows:
            if a > 0:
                SLA_limit = find_sla(find_location(data[4]), find_location(data[5]))
                pick = datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(int(float(data[1])))), "%Y-%m-%d")
                attempt_1 = datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(int(float(data[2])))), "%Y-%m-%d")
                how_many_day_1 = how_many_day(pick, attempt_1)
                if how_many_day_1 > SLA_limit:
                    writer.writerow([data[0], 1])
                else:
                    if data[3] == '':
                        writer.writerow([data[0], 0])
                    else:
                        attempt_2 = datetime.strptime(time.strftime('%Y-%m-%d', time.localtime(int(float(data[3])))), "%Y-%m-%d")
                        how_many_day_2 = how_many_day(attempt_1, attempt_2)
                        if how_many_day_2 > 3:
                            writer.writerow([data[0], 1])
                        else:
                            if (how_many_day_1 + how_many_day_2) > SLA_limit:
                                writer.writerow([data[0], 1])
                            else:
                                writer.writerow([data[0], 0])
                print(SLA_limit,a)

            a += 1
print('done!!!')
csvfile.close()
