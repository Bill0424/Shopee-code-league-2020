import datetime
import csv
Data = list()
with open("/Users/bill/Downloads/order_brush_order.csv", newline='') as data_set:
    rows = csv.reader(data_set)
    a = 0
    for i in rows:
        if a > 0:
            i[-1] = datetime.datetime.strptime(i[-1], "%Y-%m-%d %H:%M:%S")
            Data.append(i)
        a += 1

Data = sorted(Data,key = lambda s: s[1])

seperate_set = []
head = 0
tail = 0
id = Data[0][1]
i = 0
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['shopid', 'userid'])
    while True:
        max_supply = 0
        user = list()
        while Data[i][1] == id:
            i += 1
            if i >= len(Data):
                break
        tail = i-1
        seperate_set = sorted(Data[head:tail+1], key=lambda s: s[-1])
        pos = 0
        while True:
            count = dict()
            print(pos,len(seperate_set))
            time_span = seperate_set[pos][-1] + datetime.timedelta(hours=1)
            for h in seperate_set[pos:]:
                if h[-1] > time_span:
                    break
                else:
                    if h[2] not in count.keys():
                        new = {str(h[2]): 1}
                        count.update(new)
                    else:
                        count[h[2]] += 1
            rate = len(seperate_set)/len(count.keys())
            if rate >= 3:
                porpotion  = max(count.values())/len(seperate_set)
                if porpotion > max_supply:
                    max_supply = porpotion
                    user = []
                    for n in count.keys():
                        if count[n] == max(count.values()):
                            user.append(str(n))
                        else:
                            pass
                elif porpotion == max_supply:
                    for n in count.keys():
                        if count[n] == max(count.values()):
                            user.append(str(n))
                        else:
                            pass
                else:
                    pass
            pos += 1
            if seperate_set[-1][-1] <= time_span:
                break
        write_user = str
        if len(user) > 1:
            write_user = str(user[0])
            for n in user[1:]:
                write_user += "&"
                write_user += str(n)
        elif len(user) == 1:
            write_user = str(user[0])
        else:
            write_user = str(0)
        writer.writerow([seperate_set[0][1], write_user])
        head = i
        if i == len(Data):
            break
        id = Data[head][1]
