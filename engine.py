import random
import math
import pygame
import numpy as np

# 초기화
pygame.init()

# 화면 설정
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Grenade Simulation")

# 색상 정의
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (169, 169, 169)  

# 물리 상수
GRAVITY = 0.5  # 중력 가속도 
ELASTICITY = 0.6  # 탄성
FRICTION_COEFFICIENT = 0.02  # 마찰 계수 
BALL_NUM = 100 # 공 개수


#사각형 장애물
class Rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    # 충돌 처리
    def handle_collision(self, ball):
        # 수직 충돌
        if ball.x + ball.radius > self.x and ball.x - ball.radius < self.x + self.width: 
            if ball.y + ball.radius > self.y and ball.y - ball.radius < self.y + self.height:
                if ball.vel_y > 0:  # 위에서 아래로 
                    ball.y = self.y - ball.radius  # 벽 위로 위치 설정
                    ball.vel_y = -ball.vel_y * ELASTICITY  # 반사
                elif ball.vel_y < 0:  # 아래에서 위로 
                    ball.y = self.y + self.height + ball.radius  # 벽 아래로 위치 설정
                    ball.vel_y = -ball.vel_y * ELASTICITY  # 반사

        # 수평 충돌 
        if ball.y + ball.radius > self.y and ball.y - ball.radius < self.y + self.height:
            if ball.x + ball.radius > self.x and ball.x - ball.radius < self.x + self.width:
                if ball.vel_x > 0:  # 오른쪽에서 왼쪽 벽으로 충돌
                    ball.x = self.x - ball.radius  # 벽 왼쪽으로 위치 설정
                    ball.vel_x = -ball.vel_x * ELASTICITY  # 반사
                elif ball.vel_x < 0:  # 왼쪽에서 오른쪽 벽으로 충돌
                    ball.x = self.x + self.width + ball.radius  # 벽 오른쪽으로 위치 설정
                    ball.vel_x = -ball.vel_x * ELASTICITY  # 반사

        friction_force = FRICTION_COEFFICIENT * GRAVITY

        if ball.vel_x > 0:  # 오른쪽으로 이동 중
            ball.vel_x -= friction_force # 반대 방향으로 속도를 더해 마찰 효과
            if ball.vel_x < 0:  
                ball.vel_x = 0
        elif ball.vel_x < 0:  # 왼쪽으로 이동 중
            ball.vel_x += friction_force # 반대 방향으로 속도를 더해 마찰 효과
            if ball.vel_x > 0:  
                ball.vel_x = 0

# 경사형 장애물
class Slope:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1  # 경사면 시작점 x,y
        self.y1 = y1  
        self.x2 = x2  # 경사면 끝점 x,y
        self.y2 = y2  

    # 경사면의 기울기 계산
    def handle_collision(self, ball):
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        slope_angle = math.atan2(dy, dx)  # 경사면의 각도
        normal_angle = slope_angle + math.pi / 2  # 경사면에 수직인 법선 벡터

        # 공에서 경사면으로의 수직 거리 계산
        line_dx = ball.x - self.x1
        line_dy = ball.y - self.y1
        dot_product = (line_dx * dx + line_dy * dy) / (dx**2 + dy**2)
        
        # 최근접 점을 계산 
        closest_x = self.x1 + dot_product * dx
        closest_y = self.y1 + dot_product * dy
        
        # 슬로프 구간 내에 있는 점인지 확인
        if 0 <= dot_product <= 1:
            # 공과 경사면 사이의 거리 계산
            distance = math.sqrt((ball.x - closest_x)**2 + (ball.y - closest_y)**2)
            
            
            # 충돌 범위 설정 (넉넉하게)
            COLLISION_THRESHOLD = ball.radius + 2


            # 반사
            if distance < COLLISION_THRESHOLD:
                overlap = COLLISION_THRESHOLD - distance
                angle = math.atan2(ball.y - closest_y, ball.x - closest_x)  # 공과 경사면 사이의 각도
                
                # 충돌 시 공을 밀어내는 방향으로 이동
                ball.x += math.cos(angle) * overlap
                ball.y += math.sin(angle) * overlap
                
                # 경사면의 법선 벡터 계산
                normal_dx = math.cos(normal_angle)
                normal_dy = math.sin(normal_angle)
                
                # 공의 속도 벡터와 법선 벡터의 내적을 구하여 반사 계산
                dot_product = ball.vel_x * normal_dx + ball.vel_y * normal_dy
                ball.vel_x -= 2 * dot_product * normal_dx * ELASTICITY
                ball.vel_y -= 2 * dot_product * normal_dy * ELASTICITY * 1.5

# 원형 장애물         
class Circle:
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def handle_collision(self, ball):
        # 공과 원형 지형 사이의 거리 계산
        dx = ball.x - self.cx
        dy = ball.y - self.cy
        distance = math.sqrt(dx**2 + dy**2)
        
        # 공이 원에 닿았을 때
        if distance < self.radius + ball.radius:
            # 충돌 후 공이 이동할 오버랩 거리 계산
            overlap = self.radius + ball.radius - distance
            angle = math.atan2(dy, dx)  # 원과 공 사이의 각도 계산
            
            # 충돌 후 공을 밀어내는 방향으로 이동
            ball.x += math.cos(angle) * overlap
            ball.y += math.sin(angle) * overlap
            
            # 공의 속도 벡터를 원의 표면 법선 방향으로 반사
            normal_angle = angle + math.pi  # 법선 방향
            normal_dx = math.cos(normal_angle)
            normal_dy = math.sin(normal_angle)
            
            # 법선 방향으로 반사
            dot_product = ball.vel_x * normal_dx + ball.vel_y * normal_dy
            ball.vel_x -= 2 * dot_product * normal_dx
            ball.vel_y -= 2 * dot_product * normal_dy
            
            # 반사 후 속도 감소 (탄성계수 적용)
            ball.vel_x *= ELASTICITY
            ball.vel_y *= ELASTICITY
            
            # 원형 지형에 닿은 후 공이 굴러떨어지는 효과를 줌
            if abs(ball.vel_y) < 1:  # 매우 작은 y속도일 때 (수직으로 멈췄을 때)
                ball.vel_y += 1  # 중력의 효과로 공을 밀어냄



class Ball:
    def __init__(self, x, y, velx, vely,radius):
        self.x = x
        self.y = y
        self.radius = radius
        self.vel_x = velx
        self.vel_y = vely 
        self.color = BLUE
        self.angular_velocity = random.uniform(-1, 1) 

    # 좌표 업데이트
    def update(self, terrain, balls):
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

        # 회전 적용
        self.angular_velocity *= 0.99  # 공기 저항에 의한 회전 감소

        # 충돌 처리        
        for shape in terrain:
            shape.handle_collision(self)

        # 바닥에 마찰력 적용 
        if self.y + self.radius >= HEIGHT:  
            self.y = HEIGHT - self.radius 
            friction_force = FRICTION_COEFFICIENT * GRAVITY
            if self.vel_x > 0:
                self.vel_x -= friction_force
                if self.vel_x < 0:  
                    self.vel_x = 0
            elif self.vel_x < 0:
                self.vel_x += friction_force
                if self.vel_x > 0:
                    self.vel_x = 0
            self.vel_y = -self.vel_y * ELASTICITY 

        # 벽 충돌 처리
        if self.x - self.radius <= 5 or self.x + self.radius >= WIDTH-5: #가끔 벽 안쪽으로 들어가서 에러가 나 넉넉하게 설정
            self.vel_x = -self.vel_x * ELASTICITY

        if self.y - self.radius <= 5:
            self.vel_y = -self.vel_y * ELASTICITY

        # 구슬 간 충돌 처리
        for other_ball in balls:
            if other_ball != self:
                dx = self.x - other_ball.x
                dy = self.y - other_ball.y
                distance = math.sqrt(dx**2 + dy**2)
                
                # 충돌이 발생하면
                if distance < self.radius + other_ball.radius:
                    # 각도 계산
                    angle = math.atan2(dy, dx)
                    sin = math.sin(angle)
                    cos = math.cos(angle)

                    # 기존 속도를 회전시켜서 x, y 성분으로 분리
                    vel_x1 = self.vel_x * cos + self.vel_y * sin
                    vel_y1 = self.vel_y * cos - self.vel_x * sin
                    vel_x2 = other_ball.vel_x * cos + other_ball.vel_y * sin
                    vel_y2 = other_ball.vel_y * cos - other_ball.vel_x * sin

                    # 완전 탄성 충돌 처리 (속도 반사)
                    new_vel_x1 = vel_x2
                    new_vel_y1 = vel_y2
                    new_vel_x2 = vel_x1
                    new_vel_y2 = vel_y1

                    # 회전 상태 업데이트 (각속도 변화)
                    angular_velocity1 = (vel_x2 - vel_x1) * 0.5  # 속도 차이에 따른 회전 변화
                    angular_velocity2 = (vel_x1 - vel_x2) * 0.5  # 속도 차이에 따른 회전 변화

                    self.angular_velocity += angular_velocity1
                    other_ball.angular_velocity += angular_velocity2

                    # 회전된 속도를 원래 방향으로 다시 적용
                    self.vel_x = new_vel_x1 * cos - new_vel_y1 * sin
                    self.vel_y = new_vel_y1 * cos + new_vel_x1 * sin
                    other_ball.vel_x = new_vel_x2 * cos - new_vel_y2 * sin
                    other_ball.vel_y = new_vel_y2 * cos + new_vel_x2 * sin

                    overlap = self.radius + other_ball.radius - distance
                    
                    # 각 구슬을 반대 방향으로 밀어내기
                    if overlap > 0:
                        self.x += (dx / distance) * overlap / 2
                        self.y += (dy / distance) * overlap / 2
                        other_ball.x -= (dx / distance) * overlap / 2
                        other_ball.y -= (dy / distance) * overlap / 2
    


    def draw(self, surface):
        angle = self.angular_velocity * 10  # 회전된 구슬을 그릴 때 각도를 반영하여 회전시킴
        rotated_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(rotated_surface, self.color, (self.radius, self.radius), self.radius)
        rotated_surface = pygame.transform.rotate(rotated_surface, angle)
        new_rect = rotated_surface.get_rect(center=(self.x, self.y))
        surface.blit(rotated_surface, new_rect.topleft)


class Grenade:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.exploded = False
        self.timer = 0
        self.balls = []
        self.vel_x = 0
        self.vel_y = 0
        self.radius = 10  

    # 수류탄 발사
    def launch(self, angle, speed): 
        self.vel_x = math.cos(angle) * speed
        self.vel_y = math.sin(angle) * speed

    # 수류탄 폭발
    def update(self, terrain):
        if self.timer >= 120:  # 2초 후 폭발 
            self.explode()
        self.timer += 1

        # 중력 적용
        self.vel_y += GRAVITY
        self.x += self.vel_x
        self.y += self.vel_y

        for shape in terrain:
            shape.handle_collision(self)

        # 마찰력 적용 
        if self.y + self.radius >= HEIGHT:  
            self.y = HEIGHT - self.radius  
            friction_force = FRICTION_COEFFICIENT * GRAVITY
            if self.vel_x > 0:
                self.vel_x -= friction_force
                if self.vel_x < 0:  
                    self.vel_x = 0
            elif self.vel_x < 0:
                self.vel_x += friction_force
                if self.vel_x > 0:
                    self.vel_x = 0
            self.vel_y = -self.vel_y * ELASTICITY  

        # 벽과 충돌 처리
        if self.x - self.radius <= 5 or self.x + self.radius >= WIDTH-5:
            self.vel_x = -self.vel_x * ELASTICITY

        if self.y - self.radius <= 5:
            self.vel_y = -self.vel_y * ELASTICITY
    
    # 수류탄 폭발 효과
    def explode(self):
        if not self.exploded:
            self.exploded = True
            for _ in range(BALL_NUM):  # 폭발 시 N개의 구슬을 생성
                angle = random.uniform(0, 2 * math.pi)  # 랜덤 각도
                speed = random.uniform(15, 20)  # 폭발 후 구슬의 초기 속도를 크게 설정
                vel_x = math.cos(angle) * speed
                vel_y = math.sin(angle) * speed
                ball = Ball(self.x, self.y, vel_x, vel_y, 4)
                self.balls.append(ball)

    def draw(self, surface):
        if not self.exploded:
            pygame.draw.circle(surface, RED, (int(self.x), int(self.y)), self.radius)
        else:
            for ball in self.balls:
                ball.draw(surface)


def generate_terrain():
    terrain = []
    # 경사면, 장애물 생성
    terrain.append(Rectangle(100, 400, 600, 10))  # 평평한 바닥
    terrain.append(Rectangle(200, 350, 100, 20))  # 장애물
    terrain.append(Rectangle(400, 300, 100, 20))  # 장애물
    terrain.append(Rectangle(600, 250, 100, 20))  # 장애물
    terrain.append(Slope(50, 300, 250, 260))  # 경사면 예시
    terrain.append(Circle(500, 200, 50))  # 원형 장애물 예시
    
    return terrain



# 게임 루프
def game_loop():
    clock = pygame.time.Clock()
    running = True
    grenades = []  # 수류탄 목록
    terrain = generate_terrain()  # 지형 생성
    drag_start = None  # 드래그 시작점 초기화
    current_grenade = None  # 드래그 중 표시할 수류탄

    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:  # 마우스 클릭 시 드래그 시작
                if event.button == 1:  # 왼쪽 클릭
                    drag_start = pygame.mouse.get_pos()  # 드래그 시작 위치 기록
                    current_grenade = Grenade(drag_start[0], drag_start[1])  # 드래그 중 수류탄 표시

            if event.type == pygame.MOUSEBUTTONUP:  # 마우스 버튼 떼면 드래그 끝
                if event.button == 1 and drag_start:
                    drag_end = pygame.mouse.get_pos()  # 드래그 끝 위치
                    # 드래그의 방향 벡터 및 속도 계산
                    dx = drag_start[0] - drag_end[0]  # 방향이 반대이므로 빼는 순서 변경
                    dy = drag_start[1] - drag_end[1]  # 마찬가지로
                    speed = min(math.sqrt(dx**2 + dy**2) / 10, 20)  # 최대 속도 제한
                    angle = math.atan2(dy, dx)  # 반영된 방향으로 각도 계산
                    
                    grenade = Grenade(drag_start[0], drag_start[1])
                    grenade.launch_angle = angle
                    grenade.launch_speed = speed
                    
                    # 수류탄에 속도와 각도 반영
                    grenade.vel_x = math.cos(angle) * speed
                    grenade.vel_y = math.sin(angle) * speed
                    grenades.append(grenade)
                    drag_start = None
                    current_grenade = None  # 드래그 종료 후 수류탄 표시 제거


        # 마우스 드래그 동안 회색 직선 그리기
        if drag_start:
            drag_end = pygame.mouse.get_pos()  # 마우스 현재 위치
            pygame.draw.line(screen, GRAY, drag_start, drag_end, 2)

            # 드래그 중 수류탄 모양을 마우스 위치에 표시
            if current_grenade:
                current_grenade.x, current_grenade.y = drag_end  # 현재 마우스 위치로 수류탄 이동
                current_grenade.draw(screen)  # 드래그 중 수류탄을 표시

        for t in terrain:
            if isinstance(t, Rectangle):  # 사각형인 경우
                pygame.draw.rect(screen, GREEN, (t.x, t.y, t.width, t.height))
            elif isinstance(t, Slope):  # 경사면인 경우
                pygame.draw.line(screen, GREEN, (t.x1, t.y1), (t.x2, t.y2), 5)
            elif isinstance(t, Circle):  # 원형인 경우
                pygame.draw.circle(screen, GREEN, (t.cx, t.cy), t.radius)


        # 수류탄 업데이트 및 그리기
        for grenade in grenades:
            grenade.update(terrain)
            grenade.draw(screen)

        # 폭발 후 구슬들 업데이트
        for grenade in grenades:
            for ball in grenade.balls:
                ball.update(terrain, grenade.balls)  # 구슬에 지형을 반영
                ball.draw(screen)

        pygame.display.flip()
        clock.tick(60)  # FPS 60으로 설정

    pygame.quit()

# 게임 시작
game_loop()
