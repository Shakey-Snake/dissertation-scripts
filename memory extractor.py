import csv
import os

dir_path = './models/data/memory.txt'

with open("./models/data/memory_extracted.csv", 'w') as f:
    with open(dir_path, 'r') as file:
        writer = csv.writer(f)
        count = 0
        row = []
        for line in file:
            if count == 27:
                count = 0
                # print(line)
                writer.writerow(row)
                row = []
            else:
                if count == 4:
                    row.append(line[19:52])
                elif count == 11:
                    row.append('Model Size')
                    row.append(line[31:])
                elif count == 22:
                    # print(line)
                    row.append('Peak Mem')
                    row.append(line[60:])
                elif count == 24:
                    print(line)
                    row.append('VmRSS')

                    charcount = 23
                    num = line[charcount:charcount]
                    print(num)
                    while num != 'M':
                        charcount = charcount + 1
                        num = line[charcount:charcount + 1]
                        print(num)
                    row.append(line[23:charcount - 1])
                elif count == 25:
                    print(line)
                    row.append('RssAnnon')

                    charcount = 23
                    num = line[charcount:charcount]
                    print(num)
                    while num != 'M':
                        charcount = charcount + 1
                        num = line[charcount:charcount + 1]
                        print(num)
                    row.append(line[23:charcount - 1])
                elif count == 26:
                    #print(line)

                    row.append('RssFile + RssShmem')

                    charcount = 23

                    num = line[charcount:charcount]
                    print(num)
                    while num != 'M':
                        charcount = charcount + 1
                        num = line[charcount:charcount + 1]
                        # print(line)
                        print(num)
                    row.append(line[23:charcount - 1])
            count = count + 1
