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
chamada13 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_mono_faces.py\"'
chamada14 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_mono_faces_gr.py\"'
chamada15 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_bif_2.py\"'
chamada16 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python simulation_bif_2_gr.py\"'

# windows
chamada20 = 'docker run -t -it -v /c/Users/jp/Documents/git/local/Presto2:/elliptic padmec/elliptic:1.0 bash -c \"cd /elliptic; python -m  structured.cfg\"'
chamada21 = 'docker run -t -it -v $PWD:/elliptic presto bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'

chamada22 = 'sudo docker run -it -v $PWD:/elliptic presto bash -c \"cd /elliptic; python -m elliptic.Preprocess structured.cfg\"'
chamada23 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python AMS_SOLVER/tools_cy_py/setup.py\"'
chamada24 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python AMS_SOLVER/simulation_AMS.py\"'

"""
sudo docker run -it -v $PWD:/elliptic presto bash -c "cd /elliptic; python -m elliptic.Preprocess structured.cfg"
sudo docker pull padmec/elliptic:1.0
sudo docker pull padmec/pymoab-pytrilinos:3.6


"""



# print(chamada20)
# import pdb; pdb.set_trace()


l1 = [chamada1, chamada2]
l2 = [chamada11, chamada5] # monofasico
l3 = [chamada4, chamada6]
l4 = [chamada11, chamada9] # monofasico com gravidade
l5 = [chamada11, chamada12] # bifasico com gravidade
l6 = [chamada11, chamada5]
l7 = [chamada11, chamada13] # monofasico por faces
l8 = [chamada11, chamada14] # monofasico por elemento com gravidade
l9 = [chamada11, chamada15] # bifasico sem gravidade por elemento
l10 = [chamada11, chamada24] # ams_monofasico

l12 = [chamada11, chamada16] # bifasico com gravidade por elemento



# os.system(chamada5)
#
# import pdb; pdb.set_trace()


c1 = '/home/joao/Dropbox/git/Presto2.2_proj2/AMS_SOLVER/tools_cy_py'
c2 = '/home/joao/Dropbox/git/Presto2.2_proj2/presto/Preprocessors/AMS/Structured/tools_cy_py'
c3 = '/home/joao/Dropbox/git/Presto2.2_proj2'
set = 'python setup.py build_ext --inplace'

cc = [c1, c2]

for i in cc:
    os.chdir(i)
    os.system(set)

os.chdir(c3)


# for i in l10:
#     os.system(i)

os.system(chamada24)



# os.system(chamada5)

# caminho_visit = '/home/joao/programas/visit2_10_0.linux-x86_64/bin'
# os.chdir(caminho_visit)
# os.system('./visit')
