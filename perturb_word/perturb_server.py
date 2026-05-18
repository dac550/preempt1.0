"""并查集扰动服务 - 中心病名MLDP + 成员语义偏移"""
import socket
import json
import numpy as np


def call_bcv(root_orig, root_pert, child_orig, top_k=10):
    """调用 BCV 服务做类比推理"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9997))
    req = json.dumps({'root_orig': root_orig, 'root_pert': root_pert, 'child_orig': child_orig, 'top_k': top_k})
    s.sendall(req.encode('utf-8'))
    s.shutdown(socket.SHUT_WR)
    data = b''
    while True:
        chunk = s.recv(4096)
        if not chunk: break
        data += chunk
    s.close()
    return json.loads(data)


def call_mlpd(word, epsilon=5.0, domain='medical'):
    """调用 ST+FT 服务做 MLDP 扰动"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9999))
    req = json.dumps({'word': word, 'domain': domain, 'epsilon': epsilon, 'threshold_lower': 0.3, 'threshold_upper': 0.99})
    s.sendall(req.encode('utf-8'))
    s.shutdown(socket.SHUT_WR)
    data = b''
    while True:
        chunk = s.recv(4096)
        if not chunk: break
        data += chunk
    s.close()
    return json.loads(data)


def perturb_group(disease_name, members, epsilon_total=5.0, method='top1'):
    epsilon_center = epsilon_total * 0.6
    epsilon_member = epsilon_total * 0.4 / max(len(members), 1)
    
    # Step 1: 中心病名 MLDP 扰动
    result = call_mlpd(disease_name, epsilon=epsilon_center)
    if 'error' in result or not result.get('candidates'):
        return {'error': f'中心病名扰动失败: {result}'}
    new_disease = result['candidates'][0]['new_word']
    
    # Step 2: 每个成员通过 BCV 做语义偏移
    new_members = {}
    for member, mtype in members.items():
        bc = call_bcv(disease_name, new_disease, member)
        if 'error' in bc or not bc.get('results'):
            new_members[member] = {'new': member, 'method': 'keep'}
            continue
        
        if method == 'top1':
            chosen = bc['results'][0]
        else:
            sims = np.array([c['similarity'] for c in bc['results']])
            scores = epsilon_member * sims / 2.0
            scores = scores - np.max(scores)
            probs = np.exp(scores)
            probs = probs / np.sum(probs)
            idx = np.random.choice(len(bc['results']), p=probs)
            chosen = bc['results'][idx]
        
        new_members[member] = {
            'new': chosen['zh_word'],
            'sim': chosen['similarity'],
            'method': method
        }
    
    return {
        'original_disease': disease_name,
        'new_disease': new_disease,
        'epsilon_center': epsilon_center,
        'epsilon_member': epsilon_member,
        'members': new_members
    }


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9998))
server.listen(5)
print("监听 9998...", flush=True)

while True:
    conn, addr = server.accept()
    try:
        data = b''
        while True:
            chunk = conn.recv(4096)
            if not chunk: break
            data += chunk
        req = json.loads(data)
        result = perturb_group(
            req['disease'],
            req['members'],
            req.get('epsilon_total', 5.0),
            req.get('method', 'top1')
        )
        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()
