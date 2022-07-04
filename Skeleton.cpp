//=============================================================================================
// Harmadik hazi feladat: Gravitalo Gumilepedo
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Istenes Marton
// Neptun : ASDSVN
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T> g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T> g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}


typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 200;
std::vector<vec2> weights;
float m = 0.005;
vec3 g = vec3(0.0f, 0.0f, -0.00005f);

void h(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
	X = U; Y = V; Z = 0;
	for (int i = 0; i < weights.size(); i++) {
		Z = Z - (Dnum2(m + m * i) / (Pow(Pow(X - weights[i].x, 2) + Pow(Y - weights[i].y, 2), 0.5f) + 0.01));
	}
}

vec3 getNormal(float u, float v) {
	Dnum2 X, Y, Z;
	Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
	h(U, V, X, Y, Z);
	vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
	return cross(drdU, drdV);
}

float RandomFloat(float a, float b) {
	float random = (float)rand() / (RAND_MAX);
	return a + random * (b - a);
}

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z), d2(q2.x, q2.y, q2.z);
	vec3 temp(d2 * q1.w + d1 * q2.w + cross(d1, d2));
	return vec4(temp.x, temp.y, temp.z, q1.w * q2.w - dot(d1, d2));
}

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.001f; bp = 100.0f;
	}
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	mat4 P2() {
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, -2.0f / (bp - fp), 0,
			0, 0, 0, 1);
	}

};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le;
	vec4 wLightPos;
};

class CheckerBoardTexture : public Texture {

public:
	CheckerBoardTexture(const int width, const int height,vec4 c1, vec4 c2) : Texture() {
		std::vector<vec4> image(width * height);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? c1 : c2;
		}
		create(width, height, image, GL_NEAREST);
	}
};

struct RenderState {
	mat4 MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3 wEye;
};

class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

class PhongShader : public Shader {
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv;
		uniform Light[2] lights;
		uniform int   nLights;
		uniform vec3  wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[2];
		out vec2 texcoord;
		out vec3 coordinate;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
			coordinate=vtxPos;
			texcoord = vtxUV;
		}
	)";

	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[2] lights;
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;
		in  vec3 wView;
		in  vec3 wLight[2];
		in vec3 coordinate;
		in  vec2 texcoord;
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {

				vec3 lightcoor = vec3(lights[i].wLightPos.x,lights[i].wLightPos.y,lights[i].wLightPos.z);
				float d = length(lightcoor-coordinate);
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);

				float stair = 0.01 / coordinate.z;
				stair = 1 + floor(coordinate.z * 8) / 8;    
				if(coordinate.z == 0) stair = 1;

				//float stair = 1.0f;

				radiance += ka * stair * lights[i].La / pow(d,2)+ (kd * texColor * stair * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d,2);
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

struct VertexData {
	vec3 position, normal;
	vec2 texcoord;
};

class Geometry {

protected:
	unsigned int vao, vbo;
public:

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
	}
	void Load(const std::vector<VertexData>& vtxData) {
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vtxData.size() * sizeof(VertexData), &vtxData[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}
	virtual void Draw() = 0;
	virtual void Animate(float t) { }
	virtual void Update() {}
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {

	unsigned int nVtxPerStrip, nStrips;
public:

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		vtxData.texcoord = vec2(u, v);
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		Load(vtxData);
	}

	void Draw() {
		glBindVertexArray(vao);
		//for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_LINE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}

};

class Sphere : public ParamSurface {
public:
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);

		/*const float R = 1, r = 0.5f;
		U = U * 2.0f * M_PI, V = V * 2.0f * M_PI;
		Dnum2 D = Cos(U) * r + R;
		X = D * Cos(V);Y = D * Sin(V); Z = Sin(U) * r;*/
	}

};

class Plain : public ParamSurface {
public:
	Plain() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2 - 1;
		V = V * 2 - 1;
		X = U; Y = V; Z = 0;
		for (int i = 0; i < weights.size(); i++) {
			Z = Z - (Dnum2(m + m * i) / (Pow(Pow(X - weights[i].x, 2) + Pow(Y - weights[i].y, 2), 0.5f) + 0.01));
		}
	}
	void Update() {
		create();
	}

};

struct Object {
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	explicit Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
		translation = vec3(-1, -1, 0);
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		geometry->Animate(tend);
	}
};

struct Ball : public Object {

	vec3 velocity;
	vec3 force;
	vec3 bottom;
	vec3 ballup;
	bool started;

	explicit Ball(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) : Object(_shader, _material, _texture, _geometry) {
		velocity = vec3(0, 0, 0);
		bottom = vec3(-0.95, -0.95, 0);
		translation = vec3(-0.95, -0.95, 0.05);
		started = false;
	}

	void Animate(float ts, float te) {

		if (started) {
			if (bottom.x > 1) { bottom.x -= 2; }
			if (bottom.y > 1) { bottom.y -= 2; }
			if (bottom.x < -1) { bottom.x += 2; }
			if (bottom.y < -1) { bottom.y += 2; }

			vec3 n = normalize(getNormal(bottom.x, bottom.y));

			vec3 force = g - dot(g, n) * n;
			ballup = n;

			velocity = velocity + force;
			velocity = velocity - velocity * 0.001;
			velocity = velocity - dot(velocity, n) * n;

			vec4 iranyvektor(velocity.x,velocity.y,velocity.z, 1);
			vec4 forgastengely = iranyvektor * RotationMatrix(M_PI / 2, n);

			rotationAxis = vec3(forgastengely.x, forgastengely.y, forgastengely.z);
			//rotationAxis = vec3(-normalize(velocity).y, normalize(velocity).x, 0);
			//printf("%f", length(velocity));
			rotationAngle = te * length(velocity)*100;

			bottom = bottom + velocity;
			bottom.z = 0;
			for (int i = 0; i < weights.size(); i++) {
				bottom.z = bottom.z - ((m + m * i) / (sqrtf(pow((bottom.x - weights[i].x), 2) + pow((bottom.y - weights[i].y), 2)) + 0.01));
			}
			translation = bottom + n * 0.05;


		}
	}
};

class Scene {

	Camera camera;
	Object* plainobject;
	std::vector<Ball*> balls;
	std::vector<Light> lights;
	const vec4 l1 = vec4(0, 0.25, 1, 1);
	const vec4 l2 = vec4(0, -0.25, 1, 1);
	mat4 activeprojection;
	Shader* phongShader;

public:

	void Build() {

		//Texture* texture8x8 = new CheckerBoardTexture(8, 8, vec4(0.5f, 0.5f, 0.8f, 1), vec4(1, 1, 1, 1));
		Texture* texture8x8 = new CheckerBoardTexture(8, 8, vec4(1, 1, 1, 1), vec4(1, 1, 1, 1));
		activeprojection = camera.P2();

		phongShader = new PhongShader();

		Material* material = new Material;
		material->kd = vec3(1, 1, 1);
		material->ka = vec3(0.4f, 0.4f, 0.4f);
		material->ks = vec3(10, 10, 10);
		material->shininess = 100;

		
		plainobject = new Object(phongShader, material,texture8x8, new Plain());
		plainobject->translation = vec3(0, 0, 0);
		plainobject->scale = vec3(1.0f, 1.0f, 1.0f);

		camera.wEye = vec3(0, 0, 1);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		lights.resize(2);
		lights[0].wLightPos = l1;
		lights[0].La = vec3(0.1, 0.1, 0.1);
		lights[0].Le = vec3(0.5f, 0.5f, 0.8f);

		lights[1].wLightPos = l2;
		lights[1].La = vec3(0.1, 0.1, 0.1);
		lights[1].Le = vec3(0.16f, 0.0f, 0.5f);

	}

	void Render(bool o) {

		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = activeprojection;
		state.lights = lights;
		plainobject->Draw(state);

		if (o) {
			/*activeprojection = camera.P2();
			camera.wEye = vec3(10, 10, 0);
			camera.wLookat = vec3(0, 0, 0);
			camera.wVup = vec3(0, 0, 1);*/

			activeprojection = camera.P2();
			camera.wEye = vec3(0, 0, 1);
			camera.wLookat = vec3(0, 0, 0);
			camera.wVup = vec3(0, 1, 0);

			for (Ball* ball : balls) ball->Draw(state);
		}

		if (!o) {
			activeprojection = camera.P();
			if ((int)balls.size() > 1) {
				camera.wEye = balls[0]->translation;
				camera.wLookat = camera.wEye + balls[0]->velocity;
				camera.wVup = balls[0]->ballup;
			}
			if ((int)balls.size() == 1) {
				camera.wEye = vec3(-0.95, -0.95, 0.05);
				camera.wLookat = vec3(1, 1, 0);
				camera.wVup = vec3(0, 0, 1);
			}

			for (int i = 1; i < (int)balls.size(); i++) {
				if (balls.size() > 0) {
					balls[i]->Draw(state);
				}
			}
		}

	}

	void Animate(float tstart, float tend) {

		plainobject->Animate(tstart, tend);
		for (Object* ball : balls) {
			ball->Animate(tstart, tend);
		}

		float t = tend;
		vec4 q = vec4(cosf(t / 4.0f), sinf(t / 4.0f) * cosf(t) / 2.0f, sinf(t / 4) * sinf(t) / 2.0f, sinf(t / 4.0f) * sqrtf(3.0f / 4.0f));
		vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
		lights[0].wLightPos = l2 + qmul(qmul(q, l1 - l2), qinv);
		lights[1].wLightPos = l1 + qmul(qmul(q, l2 - l1), qinv);

		for (int i = 0; i < (int)balls.size(); i++) {
			if (balls.size() > 0) {
				for (int j = 0; j < (int)weights.size(); j++) {
					if (balls[i]->bottom.x <= weights[j].x + 0.03 && balls[i]->bottom.x >= weights[j].x - 0.03 && balls[i]->bottom.y <= weights[j].y + 0.03 && balls[i]->bottom.y >= weights[j].y - 0.03) {
						balls.erase(balls.begin() + i);
					}
				}
			}
		}

	}

	void addBall() {

		Texture* texture8x8 = new CheckerBoardTexture(2, 2, vec4(0, 0, 0, 1), vec4(1, 1, 1, 1));

		Material* material = new Material;
		material->kd = vec3(RandomFloat(0,1), RandomFloat(0,1), RandomFloat(0,1));
		material->ks = vec3(4, 4, 4);
		material->ka = vec3(0.1f, 0.1f, 0.1f);
		material->shininess = 1000;

		Ball* ball = new Ball(phongShader, material,texture8x8, new Sphere());
		ball->scale = vec3(0.05, 0.05, 0.05);
		balls.push_back(ball);
	}

	void startBall(float vx, float vy) {
		if (balls.size() > 0) {
			balls[(int)balls.size() - 1]->velocity = vec3(vx, vy, 0);
			balls[(int)balls.size() - 1]->started = true;
		}
	}

	void updateSurface() {
		plainobject->geometry->Update();
	}

	void Boost() {
		balls[(int)balls.size() - 1]->velocity = balls[(int)balls.size() - 1]->velocity + balls[(int)balls.size() - 1]->velocity * 0.5;
		balls[(int)balls.size() - 1]->velocity = vec3(0, 0, 0);
	}

};

Scene scene;
bool cameraswitched = false;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
	scene.addBall();
}

void onDisplay() {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render(!cameraswitched);
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 32) {
		cameraswitched = !cameraswitched;
	}
	if (key == 'c') {
		weights.clear();
		scene.updateSurface();

	}
	glutPostRedisplay();
}

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float x = ((float)pX) / (windowWidth);
		float y = -((float)pY - windowHeight) / (windowHeight);
		scene.startBall(0.005 * x, 0.005 * y);
		scene.addBall();
	}
	if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
		float w_x = (float)(pX - windowWidth / 2.0f) / (windowWidth / 2.0f);
		float w_y = -(float)(pY - windowHeight / 2.0f) / (windowHeight / 2.0f);
		weights.push_back(vec2(w_x, w_y));
		scene.updateSurface();
	}
	glutPostRedisplay();
}

void onMouseMotion(int pX, int pY) {}

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}

	glutPostRedisplay();
}
