SSV Engine — Stencil Shadow Volumes
Aplicação experimental de Computação Gráfica para estudar e avaliar Stencil Shadow Volumes, com foco no método Z-Fail.

Estado atual: a cena recebeu direção visual sci-fi, piso procedural em placas grandes, oclusores metálicos com mapas procedurais, iluminação especular, atmosfera e marcador da luz. A geração dos volumes e a passagem de stencil Z-Fail permanecem separadas dos materiais. A aparência final e os casos de borda ainda precisam ser validados em uma máquina com janela OpenGL/GPU; o ambiente headless não permite confirmar a imagem renderizada.

Escopo
Janela OpenGL e contexto gráfico com GLFW.
Renderização 3D com materiais procedurais, UVs, mipmaps, iluminação difusa/especular e neblina atmosférica.
Cenário sci-fi com piso modular, oclusores metálicos e marcador emissivo da fonte de luz.
Câmera controlável por teclado e mouse.
Depth Buffer e Stencil Buffer (solicitando pelo menos 8 bits de stencil).
Construção de volumes de sombra a partir das arestas de silhueta, com validação de malha, transformação e posição da luz.
Contagem Z-Fail com depth clamp para evitar clipping das tampas no near/far plane (pendente de validação visual em GPU).
Modo de depuração para visualizar os volumes.
Métricas de FPS e tempo por frame.
Comparação controlada com sombras ativadas e desativadas.
Tecnologias
Python 3.11 ou superior
GLFW: janela, contexto e entrada
ModernGL: gerenciamento de recursos e renderização
PyOpenGL: consulta/validação de estados OpenGL de baixo nível, incluindo bits de stencil
NumPy: operações numéricas e geometria
GLSL: shaders
A disponibilidade de um attachment stencil no framebuffer é verificada em tempo de execução. Se o contexto/driver não fornecer stencil suficiente, a aplicação informa o problema em vez de simular o resultado.

Requisito gráfico: driver compatível com OpenGL 3.3 Core, framebuffer com pelo menos 8 bits de stencil e 16 bits de depth. A aplicação solicita 8/24 bits ao criar a janela e confirma os valores efetivamente recebidos.

Executar no Windows (PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
No Windows, também é possível executar run_windows.bat para criar o ambiente virtual, instalar as dependências e iniciar o motor automaticamente.

Também pode ser executado em Linux com python3 -m venv .venv, ativação de .venv/bin/activate e os mesmos comandos de instalação.

Estrutura
src/ssv_engine/
  app.py                  # ciclo principal da aplicação
  camera/                 # câmera e controles
  graphics/               # contexto e recursos OpenGL
  metrics/                # FPS e frame time
  scene/                  # malhas, materiais, luzes e cena
    materials.py          # texturas procedurais geradas em runtime
  shadows/                # silhuetas, volumes e algoritmo Z-Fail
  shaders/                # shaders GLSL empacotados com o motor
tests/                    # testes automatizados de geometria e lógica
Etapas de implementação
Validar janela, contexto e stencil buffer.
Renderizar geometria básica com depth testing.
Implementar câmera e iluminação.
 Construir topologia de malhas fechadas e identificar silhuetas em relação à luz.
 Extrudar as silhuetas e fechar os volumes no infinito usando coordenadas homogêneas; adicionar wireframe de debug.
 Implementar a passagem Z-Fail com operações de stencil separadas para faces frontais e traseiras e composição escura nos pixels marcados.
Validar visualmente clipping, câmera dentro do volume e auto-sombreamento em GPU real.
 Adicionar wireframe de debug e métricas de FPS/frame time.
Executar experimentos reproduzíveis e documentar resultados reais.
Integridade experimental
Não registrar resultados esperados como resultados medidos. Cada experimento deverá informar hardware, driver, resolução, configuração da cena, quantidade de objetos/triângulos, FPS e frame time.

Controles da cena
W, A, S, D: mover a câmera.
Mouse: olhar ao redor.
Espaço / Shift esquerdo: subir / descer.
Setas: mover a fonte de luz no plano X/Z.
Z / X: subir / descer a luz no eixo Y (mantida acima dos cubos).
P: abrir/fechar o painel de benchmark na tela (FPS, frame time, P95, 1% low, gráfico, tempo de GPU, triângulos e resultados).
B: ligar/desligar sombras para comparação.
M: alternar o método de contagem do stencil entre Z-Fail e Z-Pass.
+ / -: aumentar/diminuir a complexidade geométrica (níveis 0–5, anéis de caixas que também projetam sombra).
O: iniciar/pausar a órbita da luz.
V: mostrar/ocultar wireframe dos volumes dos oclusores da praça.
R: restaurar câmera e luz para a posição inicial.
K (ou F5): comparação automática sombras ON x OFF (35 s medidos por condição).
J (ou F7): varredura de complexidade — ON x OFF em cada nível 0–5 (8 s por condição). K/J durante o teste cancela.
F11 ou Alt+Enter: alternar tela cheia. Esc: liberar ou capturar o cursor.
Painel de benchmark (tecla P)
O painel é desenhado dentro da janela, sem precisar abrir planilha:

Desempenho ao vivo: FPS, frame time médio, P95, pior frame e 1% low dos últimos 300 frames, com gráfico de frame time (laranja = sombras ON, azul = OFF).
Tempo de GPU por passe (queries GL_TIME_ELAPSED, lidas sem bloquear a CPU): cena opaca e passe de stencil das sombras, separadamente. Mostra também o tempo de CPU da última construção dos volumes.
Cena: método (Z-Fail/Z-Pass), sombras ON/OFF, nível de complexidade, número de objetos e oclusores, triângulos da cena, triângulos dos volumes e arestas de silhueta.
Resultados: ao fim do K, barras ON x OFF com FPS, ms, P95, 1% low e o custo percentual das sombras; ao fim do J, uma tabela por nível (objetos, triângulos de volume, FPS ON/OFF, custo) e um gráfico do custo por nível.
Os CSV/TXT continuam sendo salvos em results/benchmarks/ (agora com nível, objetos e triângulos por frame, GPU, versão OpenGL e método), mas não são necessários para ver os resultados.

O título da janela mostra FPS e tempo médio por frame. A cena contém oclusores da praça metálicos, piso texturizado, ambiente atmosférico e uma fonte pontual fixa e controlável, representada por um núcleo dourado com halo emissivo. Os mapas de piso e metal são gerados deterministicamente em runtime. O Z-Fail calcula os volumes dos oclusores da praça; V mostra o wireframe de todos eles. B alterna a aplicação das sombras para comparação. A sombra final é composta pelo teste de stencil; o wireframe é uma ferramenta de depuração geométrica.

Preparar exportação limpa
Quando a implementação estiver pronta, gere um ZIP sem ambiente virtual, caches ou arquivos temporários:

python scripts/export_project.py --dry-run  # conferir os arquivos
python scripts/export_project.py           # gerar o ZIP
O arquivo será criado em dist/ssv-engine-source.zip. Para executar o projeto exportado no Windows, crie um ambiente virtual, instale requirements.txt e execute python main.py. Os CSVs de benchmark serão gravados em results/benchmarks/; essa pasta é excluída automaticamente do ZIP de código-fonte.

Verificação do projeto
Execute os testes, a checagem de sintaxe e a conferência do manifesto de exportação com um único comando:

.\.venv\Scripts\Activate.ps1
python scripts/check_project.py
Os testes não substituem a validação visual do OpenGL, que será feita depois da exportação no Windows.

Para executar somente os testes:

python -m unittest discover -s tests -v
