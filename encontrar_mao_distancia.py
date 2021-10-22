# 1. Importar as bibliotecas
import cv2
from cvzone.HandTrackingModule import HandDetector

# 2. Carregar o módulo de detecção
detector = HandDetector(maxHands=2, detectionCon=0.8, minTrackCon=0.8)

# 3. Caputra de vídeo
cap = cv2.VideoCapture(0)

while True:
    # Detectar as mãos
    sucesso, imagem = cap.read()
    maos, imagem = detector.findHands(imagem)
    
    # Extrair as informações das mãos
    if maos:
        # Mão 1
        mao_1 = maos[0]
        lista_landmark_1 = mao_1['lmList']  # lista dos 21 landmarks
        caixa_limite_1 = mao_1['bbox']  # informações de x, y, largura e altura da caixa delimitadora
        ponto_central_1 = mao_1['center']  # fornece o cx e cy da mão, o meio da palma
        tipo_mao_1 = mao_1['type']  # indica se é a mão esquerda ou direita
        
        # Detectar se os dedos estão para cima da mao_1
        dedos_1 = detector.fingersUp(mao_1)
        
        # Mão 2
        if len(maos) == 2:
            mao_2 = maos[1]
            lista_landmark_2 = mao_2['lmList']  # lista dos 21 landmarks
            caixa_limite_2 = mao_2['bbox']  # informações de x, y, largura e altura da caixa delimitadora
            ponto_central_2 = mao_2['center']  # fornece o cx e cy da mão, o meio da palma
            tipo_mao_2 = mao_2['type']  # indica se é a mão esquerda ou direita
            
            # Detectar se os dedos estão para cima da mao_2
            dedos_2 = detector.fingersUp(mao_2)
            
            # Encontrar a distância entre as mãos
            comprimento, info, imagem = detector.findDistance(ponto_central_1, ponto_central_2, imagem)

    # 3.3. Mostrar a imagem na tela
    cv2.imshow('Imagem', imagem)

    # 3.4. Terminar o loop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# 4. Fechar a tela de captura
cap.release()
cv2.destroyAllWindows()