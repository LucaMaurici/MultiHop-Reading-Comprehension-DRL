import os

os.system('cmd /c "netstat -ano | findstr :9000 > prova.txt"')

f = open("prova.txt", "r")

var_lettura = f.read().split()

print(var_lettura)

pid = var_lettura[4]

f.close()

os.system('cmd /c "del prova.txt"')

cmd = 'cmd /c "taskkill /f /pid ' + pid + '"'

os.system(cmd)