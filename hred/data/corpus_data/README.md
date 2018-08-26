# 세종 코퍼스 대화 데이터
수집한 데이터 중 강연등과 같은 데이터는 제외하고 대화로 구성된 데이터만 다루었습니다. (총 89개)

## processed_data_without_colon
각 대화 파일마다 텍스트만 가져온 파일  
파일 하나 당 하나의 주제로 이루어져 있음

## two_people.txt & two_people_even.txt
화자가 두명인 대화 자료들을 모아놓은 파일  
각 주제 덩어리를 \n\n으로 구분  
two_people_even.txt는 대화가 홀수 개의 줄로 이루어져 있으면 마지막 한 줄을 제거한 파일

## data_information.txt
processed_data_without_colon의 각 파일들이 몇명의 화자, 총 몇 줄의 대화로 이루어져 있는지에 대한 정보를 담고 있음
