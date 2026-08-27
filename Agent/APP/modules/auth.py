import streamlit as st
import sqlite3
import pandas as pd
import bcrypt
import datetime
import os

# 数据库路径 (自动存放在项目根目录)
DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'beaver_users.db')

def init_db():
    """初始化数据库"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash BLOB,
            role TEXT,
            signup_time TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def register_user(username, password):
    """注册逻辑"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # 默认第一个注册的人是 admin，之后的都是 user
        c.execute('SELECT count(*) FROM users')
        user_count = c.fetchone()[0]
        role = 'admin' if user_count == 0 else 'user'
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        c.execute('INSERT INTO users VALUES (?, ?, ?, ?)', 
                  (username, hashed, role, datetime.datetime.now()))
        conn.commit()
        return True, "注册成功，请登录！"
    except sqlite3.IntegrityError:
        return False, "用户名已存在！"
    finally:
        conn.close()

def login_user(username, password):
    """登录逻辑"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT password_hash, role FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    
    if data:
        if bcrypt.checkpw(password.encode('utf-8'), data[0]):
            return True, data[1] # 返回 (Success, Role)
    return False, None

def get_all_users():
    """管理员查看所有用户"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT username, role, signup_time FROM users", conn)
    conn.close()
    return df

def render_login_interface():
    """渲染登录/注册页面"""
    st.markdown("## 🔐 BEAVER Platform 访问权限")
    
    tab1, tab2 = st.tabs(["登录", "注册新账号"])
    
    with tab1:
        with st.form("login_form"):
            user = st.text_input("用户名")
            pw = st.text_input("密码", type="password")
            submitted = st.form_submit_button("进入平台")
            if submitted:
                success, role = login_user(user, pw)
                if success:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = user
                    st.session_state['role'] = role
                    st.rerun()
                else:
                    st.error("账号或密码错误")

    with tab2:
        with st.form("register_form"):
            # === 修改点在这里 ===
            # placeholder: 显示在输入框里的灰色提示文字，用户输入时会自动消失
            # help: 鼠标悬停时显示的小问号提示
            new_user = st.text_input(
                "设置用户名", 
                placeholder="高校+课题组+姓名", 
                help="为了方便后台统计，请按照格式填写，例如：东华大学+A组+张三"
            )
            new_pw = st.text_input("设置密码", type="password")
            
            reg_submitted = st.form_submit_button("注册")
            if reg_submitted:
                if new_user and new_pw:
                    # 这里加了一个简单的非强制检查（可选），如果用户没填 "+" 也可以注册，
                    # 但你可以根据需要开启下面的 strict 检查
                    # if "+" not in new_user:
                    #     st.warning("请按照格式填写：高校+课题组+姓名")
                    # else: ...
                    
                    success, msg = register_user(new_user, new_pw)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    st.warning("请输入用户名和密码")