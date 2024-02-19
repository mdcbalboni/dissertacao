# Dissertação

Todos os arquivos devem ter o caminho modificado para o diretório desejado. Este é o único parâmetro que precisa ser alterado obrigatoriamente, caso contrário, podem ocorrer erros.

## Pasta "scripts"

Nesta pasta, encontram-se os scripts necessários para realizar as predições.

- **cria_dataset.py**: Gera o conjunto de dados com base nas funções de escore desejadas. Para isso, é necessário colocar todos os conjuntos de dados das funções de escore em uma mesma pasta e, dentro do arquivo .py, especificar esse caminho como parâmetro.

- **pre_processamento.py**: Realiza o pré-processamento, normalização e alguma limpeza nos dados.

- **prepara_todos_ligantes.py**: Converte todos os arquivos .mol2 ou .sdf para arquivo .pdbqt.

- **prepara_todas_proteinas.py**: Converte todos os arquivos .pdb para arquivo .pdbqt.

- **resultados.py**: Realiza o treinamento do modelo e gera os gráficos.

## Pasta "funcao_de_escore"

Contém as nove funções de escore utilizadas no presente trabalho. Cada pasta possui um automatizador dessas funções. Para executar, basta chamar o arquivo `gera_funcao.py`. As funções de machine learning têm uma flag que, quando ativada, gera o conjunto de dados. Normalmente, essa etapa demora algumas horas para ser concluída, então a flag é desativada por padrão. Caso deseje gerar o conjunto de dados, basta modificar essa flag.

## Arquivos Pesados

Os arquivos mais pesados podem ser encontrados no seguinte link: [Google Drive](https://drive.google.com/drive/folders/1uGT7S-xFXEVLJnDlDkxjpPxFzxicrzvj?usp=sharing).

Para esclarecimento de dúvidas, entre em contato por e-mail: baalbis@gmail.com.
