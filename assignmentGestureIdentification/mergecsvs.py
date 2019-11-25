import os

root = '/Users/renil.joseph/Documents/github/class/mcCode/assign2/10'
roott = '/Users/renil.joseph/Documents/github/class/mcCode/assign2'
dirs = os.listdir(root)
for d in dirs:
    if d.startswith('.'):
        continue
    folders = os.listdir(root+'/'+d+'/Signs/videos/')
    for f in folders:
        if f.startswith('.'):
            continue
        name = f.split('_')[0]
        typevar = name
        name  = name+'.csv'
        newn = roott+'/csv/'+name
        if not os.path.exists(newn):
            output = open(newn, 'w+')
        else:
            output = open(newn, 'a')
        inputf = open(root+'/'+d+'/Signs/videos/'+f+'/key_points.csv')
        bol = True
        if os.path.getsize(newn) > 0:
            inputf.next()
            bol = False
        for line in inputf:
            line = line.strip('\n')
            if bol:
                line = line + ',type\n'
            else:
                line = line+','+typevar+'\n'
            output.write(line)
            bol = False
        inputf.close()
        output.close()


# create single csv
# import os
# files = os.listdir('.')
# output = open('/Users/renil.joseph/Documents/github/class/mcCode/assign2/data.csv', 'w+')
# input = open('/Users/renil.joseph/Documents/github/class/mcCode/assign2/csv/LIP.csv', 'r')
# row = input.next()
# output.write(row)
# input.close()

# for filenames in files:
#     input = open(filenames, 'r')
#     input.next()
#     for line in input:
#         output.write(line)
#     input.close()
# output.close()
#     