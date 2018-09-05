import os
# import platform
# print(platform.system())
# print(platform.release())

principal = '/home/joao/git/back2/Presto2'

chamada1 = 'sudo systemctl start docker'
chamada2 = 'su -c \'setenforce 0\''
chamada4 = 'sudo docker run -v $PWD:/elliptic padmec/elliptic:1.0 bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'
chamada5 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_mono.py\"'
chamada6 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_bif.py\"'
chamada7 = 'python fazer_graficos_mono.py'
chamada8 = 'python fazer_graficos_bif.py'
chamada9 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python script_gravidade_mono.py\"'
chamada10 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python -m Preprocessor.py structured.cfg\"'
chamada11 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python -m Preprocess structured.cfg\"'
chamada12 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python script_gravidade_bif.py\"'

# windows
chamada20 = 'docker run -t -it -v /c/Users/jp/Documents/git/local/Presto2:/elliptic padmec/elliptic:1.0 bash -c \"cd /elliptic; python -m  structured.cfg\"'
chamada21 = 'docker run -t -it -v $PWD:/elliptic presto bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'

chamada22 = 'sudo docker run -it -v $PWD:/elliptic presto bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'

"""
sudo docker run -it -v $PWD:/elliptic presto bash -c "cd /elliptic; python -m elliptic.Preprocess structured.cfg"
sudo docker pull padmec/elliptic:1.0
sudo docker pull padmec/pymoab-pytrilinos:3.6


"""



# print(chamada20)
# import pdb; pdb.set_trace()


l1 = [chamada1, chamada2]
l2 = [chamada4, chamada5]
l3 = [chamada4, chamada6]
l4 = [chamada11, chamada9]
l5 = [chamada11, chamada12] # bifasico com gravidade
l6 = [chamada11, chamada5]


# os.chdir(principal)
# os.system(chamada5)
#
for i in l6:
    os.system(i)

# os.system(chamada9)



# os.system(chamada5)

# caminho_visit = '/home/joao/programas/visit2_10_0.linux-x86_64/bin'
# os.chdir(caminho_visit)
# os.system('./visit')
