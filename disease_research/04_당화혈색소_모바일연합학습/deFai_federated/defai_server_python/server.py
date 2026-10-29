#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일 레이어 HbA1c 예측 연합학습 서버
레이어 1개, 노드 1개로 구성된 가장 간단한 버전
"""

import json
import numpy as np
import tensorflow as tf
import datetime
import os
from pathlib import Path
from flask import Flask, request, jsonify
import threading
import time

# Flask 앱 생성
app = Flask(__name__)

# 저장 디렉토리: 현재 스크립트 폴더 하위로 저장
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_models")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class SimpleHbA1cModel:
    """단일 레이어 HbA1c 예측 모델 (레이어 1개, 노드 1개)"""
    
    def __init__(self):
        # 가장 간단한 모델: 입력 2개 -> 출력 1개 (레이어 1개, 노드 1개)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, activation='linear', input_shape=(2,), name='single_layer')
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.build(input_shape=(None, 2))
        
        # 파라미터 정보 출력
        weights = self.model.get_weights()
        print(f"\n🔧 단일 레이어 HbA1c 모델 생성:")
        print(f"   총 파라미터: {sum(w.size for w in weights)}개")
        for i, w in enumerate(weights):
            print(f"   레이어 {i}: {w.shape} = {w.size}개")
    
    def get_parameters(self):
        """모델 파라미터를 JSON 직렬화 가능한 형태로 반환"""
        weights = self.model.get_weights()
        params = []
        for i, weight in enumerate(weights):
            param_dict = {
                'layer': i,
                'shape': list(weight.shape),
                'data': weight.tolist(),  # numpy -> list
                'size': weight.size
            }
            params.append(param_dict)
        return params
    
    def set_parameters(self, params):
        """JSON 파라미터를 모델에 설정"""
        weights = []
        for param in params:
            # list -> numpy 배열로 변환
            weight = np.array(param['data'], dtype=np.float32)
            
            # 1차원 배열을 원래 shape으로 재구성
            if len(param['shape']) > 1:
                # 2차원 이상인 경우 reshape
                weight = weight.reshape(param['shape'])
            
            weights.append(weight)
        
        self.model.set_weights(weights)
        print(f"✅ 모델 파라미터 업데이트 완료: {len(weights)}개 레이어")
    
    def train(self, epochs=3):
        """모델 훈련 - 서버에서는 실제 훈련하지 않음"""
        print("⚠️ 서버에서는 실제 훈련을 수행하지 않습니다.")
        print("   실제 훈련은 각 클라이언트에서 수행됩니다.")
        
        return {
            'loss': 0.0,
            'num_examples': 0,
            'parameters': self.get_parameters()
        }
    
    def evaluate(self):
        """모델 평가 - 서버에서는 기본 정보만 반환"""
        return {
            'loss': 0.0,
            'mae': 0.0,
            'num_examples': 0,
            'message': '서버는 파라미터 집계/배포만 담당합니다.'
        }

# 전역 변수
global_model = SimpleHbA1cModel()
current_round = 0
session_id = 1
total_rounds = 5
client_parameters = []
round_results = []

def save_parameters(parameters, round_num, session_id):
    """파라미터를 파일로 저장 (JSON + .npy)"""
    try:
        session_dir = os.path.join(SAVE_DIR, f"session_{session_id}")
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        print(f"🔧 파라미터 저장 시작: 세션 {session_id}, 라운드 {round_num}")
        print(f"   📁 저장 디렉토리: {session_dir}")
        print(f"   📊 파라미터 개수: {len(parameters)}개")
        
        # 1. JSON 형태로 저장 (메타데이터 포함)
        param_file = os.path.join(session_dir, f"round_{round_num}_parameters.json")
        with open(param_file, 'w') as f:
            json.dump(parameters, f, indent=2)
        
        # 2. .npy 파일로도 저장 (실제 값들)
        saved_files = []
        for i, param in enumerate(parameters):
            data = np.array(param['data'])
            filename = f"round_{round_num}_layer_{i}.npy"
            filepath = os.path.join(session_dir, filename)
            np.save(filepath, data)
            saved_files.append(filename)
            print(f"   💾 레이어 {i} 저장: {filename} (shape: {param['shape']})")
        
        # 3. 메타데이터 저장
        metadata = {
            'round': round_num,
            'session_id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_params': sum(p['size'] for p in parameters),
            'num_layers': len(parameters),
            'layer_shapes': [p['shape'] for p in parameters],
            'saved_files': saved_files
        }
        
        meta_file = os.path.join(session_dir, f"round_{round_num}_metadata.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 파라미터 저장 완료: 세션 {session_id}, 라운드 {round_num}")
        print(f"   📂 저장된 파일: {len(saved_files)}개")
        print(f"   📊 총 파라미터: {metadata['total_params']}개")
        
        return True
    except Exception as e:
        print(f"❌ 파라미터 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_parameters(round_num, session_id):
    """저장된 파라미터를 로드 (JSON 또는 .npy에서)"""
    try:
        session_dir = os.path.join(SAVE_DIR, f"session_{session_id}")
        
        # 1. JSON 파일에서 로드 시도
        param_file = os.path.join(session_dir, f"round_{round_num}_parameters.json")
        if os.path.exists(param_file):
            print(f"🔧 파라미터 로드 시작: 세션 {session_id}, 라운드 {round_num}")
            print(f"   📁 로드 디렉토리: {session_dir}")
            
            with open(param_file, 'r') as f:
                parameters = json.load(f)
            
            print(f"✅ 파라미터 로드 완료: 세션 {session_id}, 라운드 {round_num}")
            print(f"   📊 총 파라미터: {sum(p['size'] for p in parameters)}개")
            return parameters
        
        # 2. .npy 파일에서 로드 시도
        metadata_file = os.path.join(session_dir, f"round_{round_num}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            num_layers = metadata['num_layers']
            layer_shapes = metadata['layer_shapes']
            
            print(f"🔧 .npy 파일에서 파라미터 로드: 세션 {session_id}, 라운드 {round_num}")
            print(f"   📊 로드할 레이어 수: {num_layers}개")
            
            parameters = []
            for i in range(num_layers):
                filename = f"round_{round_num}_layer_{i}.npy"
                filepath = os.path.join(session_dir, filename)
                
                if os.path.exists(filepath):
                    data = np.load(filepath)
                    param = {
                        'layer': i,
                        'shape': layer_shapes[i],
                        'data': data.tolist(),
                        'size': data.size
                    }
                    parameters.append(param)
                    print(f"   📂 레이어 {i} 로드: {filename} (shape: {data.shape})")
                else:
                    print(f"   ⚠️ 파일 없음: {filename}")
                    return None
            
            print(f"✅ 파라미터 로드 완료: 세션 {session_id}, 라운드 {round_num}")
            return parameters
        
        print(f"⚠️ 저장된 파라미터 없음: 세션 {session_id}, 라운드 {round_num}")
        return None
        
    except Exception as e:
        print(f"❌ 파라미터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def aggregate_parameters(client_params_list):
    """클라이언트 파라미터들을 평균으로 집계"""
    if not client_params_list:
        return None
    
    # 첫 번째 클라이언트의 파라미터 구조를 기준으로 함
    base_params = client_params_list[0]
    aggregated_params = []
    
    print(f"🔧 파라미터 집계 시작: {len(client_params_list)}개 클라이언트")
    
    for layer_idx in range(len(base_params)):
        # 모든 클라이언트의 해당 레이어 파라미터 수집
        layer_params = []
        for client_params in client_params_list:
            if layer_idx < len(client_params):
                layer_data = np.array(client_params[layer_idx]['data'])
                layer_params.append(layer_data)
        
        if layer_params:
            # 평균 계산
            avg_param = np.mean(layer_params, axis=0)
            # 원래 shape을 클라이언트에서 받아서 그대로 사용
            original_shape = base_params[layer_idx]['shape']
            aggregated_param = {
                'layer': layer_idx,
                'shape': original_shape,  # 반드시 원래 shape으로!
                'data': avg_param.tolist(),
                'size': avg_param.size
            }
            aggregated_params.append(aggregated_param)
            
            print(f"   레이어 {layer_idx}: {len(layer_params)}개 클라이언트 → 평균")
    
    print(f"✅ 파라미터 집계 완료: {len(aggregated_params)}개 레이어")
    return aggregated_params

def list_saved_sessions():
    """저장된 세션 목록 조회"""
    if not os.path.exists(SAVE_DIR):
        print("📁 저장된 세션이 없습니다.")
        return []
    
    sessions = []
    for item in os.listdir(SAVE_DIR):
        if item.startswith("session_"):
            session_id = int(item.split("_")[1])
            session_dir = os.path.join(SAVE_DIR, item)
            
            # 해당 세션의 라운드 수 확인
            rounds = []
            for file in os.listdir(session_dir):
                if file.startswith("round_") and file.endswith("_metadata.json"):
                    round_num = int(file.split("_")[1])
                    rounds.append(round_num)
            
            sessions.append({
                'session_id': session_id,
                'rounds': sorted(rounds),
                'total_rounds': len(rounds)
            })
    
    return sessions

def print_saved_sessions():
    """저장된 세션 정보 출력"""
    sessions = list_saved_sessions()
    
    if not sessions:
        print("📁 저장된 세션이 없습니다.")
        return
    
    print("\n📋 저장된 세션 목록:")
    print("=" * 50)
    for session in sessions:
        print(f"   세션 {session['session_id']}: {session['total_rounds']}라운드")
        print(f"      라운드: {session['rounds']}")
    print("=" * 50)

# Flask 라우트들
@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.datetime.now().isoformat(),
        'model_info': {
            'total_params': sum(p['size'] for p in global_model.get_parameters()),
            'num_layers': len(global_model.get_parameters())
        }
    })

@app.route('/get_parameters', methods=['GET'])
def get_parameters():
    """현재 모델 파라미터 반환"""
    params = global_model.get_parameters()
    
    print(f"📤 파라미터 전송 (라운드 {current_round}):")
    for i, param in enumerate(params):
        data = np.array(param['data'])
        print(f"   레이어 {i}: {param['shape']} = {param['size']}개")
        flat = data.flatten()
        if flat.size <= 10:
            print(f"     values: {flat}")
        else:
            print(f"     first 5: {flat[:5]}")
            print(f"     last 5: {flat[-5:]}")
    
    return jsonify({
        'status': 'success',
        'round': current_round,
        'parameters': params
    })

@app.route('/submit_parameters', methods=['POST'])
def submit_parameters():
    """클라이언트로부터 파라미터 수신"""
    global current_round, global_model
    try:
        data = request.get_json()
        client_params = data.get('parameters', [])
        client_id = data.get('client_id', 'unknown')

        print(f"📥 클라이언트 {client_id} 파라미터 수신 (라운드 {current_round}):")
        for i, param in enumerate(client_params):
            print(f"   레이어 {i}: {param['shape']} = {param['size']}개")

        # 클라이언트 파라미터 저장
        client_parameters.append({
            'client_id': client_id,
            'parameters': client_params,
            'timestamp': datetime.datetime.now().isoformat()
        })

        # 자동 집계 수행 (단일 클라이언트 가정: 쌓인 모든 제출을 평균)
        client_params_list = [cp['parameters'] for cp in client_parameters]
        aggregated_params = aggregate_parameters(client_params_list)

        if not aggregated_params:
            return jsonify({'status': 'error', 'message': '집계 실패'}), 500

        # 글로벌 모델 업데이트 및 저장
        print("[서버] 자동 집계 후 set_parameters 전달값:")
        for i, param in enumerate(aggregated_params):
            print(f"  레이어 {i}: layer={param['layer']} shape={param['shape']} size={param['size']}")
        global_model.set_parameters(aggregated_params)

        save_success = save_parameters(aggregated_params, current_round, session_id)
        if not save_success:
            print(f"❌ 라운드 {current_round} 파라미터 저장 실패")
            return jsonify({'status': 'error', 'message': '파라미터 저장 실패'}), 500

        # 라운드 결과 저장 및 상태 업데이트
        round_results.append({
            'round': current_round,
            'num_clients': len(client_parameters),
            'timestamp': datetime.datetime.now().isoformat()
        })

        print(f"\n📊 라운드 {current_round} 자동 집계 완료:")
        print(f"   참여 클라이언트: {len(client_parameters)}개")
        print(f"   집계된 파라미터: {len(aggregated_params)}개 레이어")

        client_parameters.clear()
        current_round += 1

        return jsonify({
            'status': 'success',
            'message': f'라운드 {current_round-1} 자동 집계 및 저장 완료',
            'next_round': current_round
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/aggregate', methods=['POST'])
def aggregate():
    """파라미터 집계 수행"""
    global current_round, global_model
    
    try:
        if not client_parameters:
            return jsonify({
                'status': 'error',
                'message': '집계할 클라이언트 파라미터가 없습니다.'
            }), 400
        
        # 파라미터 집계
        client_params_list = [cp['parameters'] for cp in client_parameters]
        aggregated_params = aggregate_parameters(client_params_list)
        
        if aggregated_params:
            print("[서버] set_parameters 전달값:")
            for i, param in enumerate(aggregated_params):
                print(f"  레이어 {i}: layer={param['layer']} shape={param['shape']} size={param['size']} data_type={type(param['data'])}")
                if isinstance(param['data'], list):
                    print(f"    data 샘플: {param['data'][:5]}")
            # 글로벌 모델 업데이트
            global_model.set_parameters(aggregated_params)
            
            # 파라미터 저장
            save_success = save_parameters(aggregated_params, current_round, session_id)
            if not save_success:
                print(f"❌ 라운드 {current_round} 파라미터 저장 실패")
                return jsonify({
                    'status': 'error',
                    'message': '파라미터 저장 실패'
                }), 500
            
            # 라운드 결과 저장
            round_results.append({
                'round': current_round,
                'num_clients': len(client_parameters),
                'timestamp': datetime.datetime.now().isoformat()
            })
            
            print(f"\n📊 라운드 {current_round} 집계 완료:")
            print(f"   참여 클라이언트: {len(client_parameters)}개")
            print(f"   집계된 파라미터: {len(aggregated_params)}개 레이어")
            
            # 집계된 파라미터 상세 출력
            for i, param in enumerate(aggregated_params):
                data = np.array(param['data'])
                print(f"   ─ 레이어 {i} ─────────────────────────────")
                print(f"     shape: {param['shape']}")
                print(f"     min: {np.min(data):.6f}, max: {np.max(data):.6f}")
                print(f"     mean: {np.mean(data):.6f}, std: {np.std(data):.6f}")
                flat = data.flatten()
                if flat.size <= 10:
                    print(f"     values: {flat}")
                else:
                    print(f"     first 5: {flat[:5]}")
                    print(f"     last 5: {flat[-5:]}")
            
            # 클라이언트 파라미터 초기화
            client_parameters.clear()
            current_round += 1
            
            return jsonify({
                'status': 'success',
                'message': f'라운드 {current_round-1} 집계 완료',
                'aggregated_parameters': aggregated_params,
                'next_round': current_round
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/train', methods=['POST'])
def train():
    """서버에서 모델 훈련"""
    try:
        data = request.get_json()
        epochs = data.get('epochs', 3)
        
        result = global_model.train(epochs)
        
        return jsonify({
            'status': 'success',
            'training_result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate', methods=['GET'])
def evaluate():
    """모델 평가"""
    try:
        result = global_model.evaluate()
        
        return jsonify({
            'status': 'success',
            'evaluation_result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """현재 상태 정보"""
    return jsonify({
        'current_round': current_round,
        'total_rounds': total_rounds,
        'session_id': session_id,
        'connected_clients': len(client_parameters),
        'round_results': round_results
    })

def print_banner():
    """서버 시작 배너"""
    print("=" * 60)
    print("🌸 단일 레이어 HbA1c 예측 연합학습 서버 🌸")
    print("포트: 8001")
    print(f"📅 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💾 저장 디렉토리: {SAVE_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    print_banner()
    
    # 저장된 세션 정보 출력
    print_saved_sessions()
    
    print(f"\n🚀 서버 시작 중... (포트: 8001)")
    print("   클라이언트 연결 대기 중...")
    print("   서버는 계속 실행됩니다. 종료하려면 Ctrl+C를 누르세요.")
    
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=8001, debug=False)