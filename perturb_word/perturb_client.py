"""并查集扰动测试"""
import json
import socket


def query(disease, members, epsilon_total=5.0, method='top1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 9998))
    req = json.dumps({
        'disease': disease,
        'members': members,
        'epsilon_total': epsilon_total,
        'method': method
    })
    s.sendall(req.encode('utf-8'))
    s.shutdown(socket.SHUT_WR)
    data = b''
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    s.close()
    return json.loads(data)


# ===== 测试并查集 =====
test_groups = [
    {
        'disease': '肺炎',
        'members': {'头孢曲松': 'drug', '咳嗽': 'symptom'}
    },
    {
        'disease': '糖尿病',
        'members': {'二甲双胍': 'drug', '多饮': 'symptom', '甜食': 'forbid'}
    },
    {
        'disease': '高血压',
        'members': {'硝苯地平': 'drug', '头晕': 'symptom'}
    },
    {
        'disease': '哮喘',
        'members': {'沙丁胺醇': 'drug', '喘息': 'symptom'}
    }
]

for g in test_groups:
    print(f"\n{'='*60}")
    print(f"中心病名: {g['disease']}")
    print(f"成员: {g['members']}")

    r = query(g['disease'], g['members'], epsilon_total=5.0, method='top1')

    if 'error' in r:
        print(f"❌ {r['error']}")
    else:
        print(f"新病名: {r['new_disease']}")
        print(f"隐私预算: 中心ε={r['epsilon_center']:.1f}, 成员ε={r['epsilon_member']:.2f}")
        print(f"扰动结果:")
        for orig, info in r['members'].items():
            print(f"  {orig} → {info['new']} (sim={info.get('sim', 0):.3f}, method={info.get('method', '?')})")

print(f"\n✅ 完成")
