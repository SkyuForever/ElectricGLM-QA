## 第一章 函数与极限

初等数学的研究对象基本上是不变的量, 而高等数学的研究对象则是变动 的量. 所谓函数关系就是变量之间的依赖关系, 极限方法是研究变量的一种基本 方法. 本章将介绍映射、函数、极限和函数的连续性等基本概念以及它们的一些 性质。

## 第一节 映射与函数

## 一、集合

## 1. 集合概念

集合是数学中的一个基本概念, 我们先通过例子来说明这个概念. 例如, 一 个书柜中的书构成一个集合, 一间教室里的学生构成一个集合, 全体实数构成一 个集合等等.一般的, 所谓集合(简称集)是指具有某种特定性质的事物的总体, 组成这个集合的事物称为该集合的元素(简称元).

通常用大写拉丁字母 A, B, C, $\cdots$ 表示集合,用小写拉丁字母 $a, b, c, \cdots$ 表 示集合的元素. 如果 $a$ 是集合 $A$ 的元素, 就说 $a$ 属于 $A$, 记作 $a \in A$; 如果 $a$ 不 是集合 $A$ 的元素, 就说 $a$ 不属于 $A$, 记作 $a \notin A$ 或 $a \bar{\in} A$.一个集合, 若它只含 有限个元素, 则称为有限集; 不是有限集的集合称为无限集.

表示集合的方法通常有以下两种: 一种是列举法, 就是把集合的全体元索一 一列举出来表示.例如, 由元素 $a_{1}, a_{2}, \cdots, a_{n}$ 组成的集合 $A$, 可表示成

$$
A=\left\{a_{1}, a_{2}, \cdots, a_{n}\right\} ;
$$

另一种是描述法,若集合 $M$ 是由具有某种性质 $P$ 的元素 $x$ 的全体所组成的, 就 可表示成

$$
M=\{x \mid x \text { 具有性质 } P\} \text {. }
$$

例如,集合 $B$ 是方程 $x^{2}-1=0$ 的解集, 就可表示成

$$
B=\left\{x \mid x^{2}-1=0\right\} .
$$

对于数集,有时我们在表示数集的字母的右上角标上“*”来表示该数集内 排除 0 的集,标上“+”来表示该数集内排除 0 与负数的集.

习惯上,全体非负整数即自然数的集合记作 $\mathrm{N}$, 即

$$
\mathbf{N}=\{0,1,2, \cdots, n, \cdots\}
$$

全体正整数的集合为

$$
\mathbf{N}^{+}=\{1,2,3, \cdots, n, \cdots\} ;
$$

全体整数的集合记作 $\mathbf{Z}$, 即

$$
\mathrm{Z}=\{\cdots,-n, \cdots,-2,-1,0,1,2, \cdots, n, \cdots\} ;
$$

全体有理数的集合记作 $\mathbf{Q}$; 即

$$
\mathbf{Q}=\left\{\frac{p}{q} \mid p \in \mathbf{Z}, q \in \mathbf{N}^{+} \text {且 } p \text { 与 } q \text { 互质 }\right\} ;
$$

全体实数的集合记作 $\mathbf{R}, \mathbf{R}^{*}$ 为排除数 0 的实数集, $\mathbf{R}^{+}$为全体正实数的集.

. 设 $A 、 B$ 是两个集合,如果集合 $A$ 的元素都是集合 $B$ 的元素,则称 $A$ 是 $B$ 的子集,记作 $A \subset B$ (读作 $A$ 包含于 $B$ ) 或 $B \supset A$ (读作 $B$ 包含 $A$ ).

如果集合 $A$ 与集合 $B$ 互为子集, 即 $A \subset B$ 且 $B \subset A$, 则称集合 $A$ 与集合 $B$ 相等, 记作 $A=B$. 例如, 设

$$
A=\{1,2\}, \quad B=\left\{x \mid x^{2}-3 x+2=0\right\},
$$

则 $A=B$.

若 $A \subset B$ 且 $A \neq B$, 则称 $A$ 是 $B$ 的真子集，记作 $A \varsubsetneqq B$. 例如, $\mathbf{N} \varsubsetneqq \mathbf{Z} \varsubsetneqq \mathbf{Q} \subsetneq \mathbf{R}$. 不含任何元素的集合称为空焦.例如

$$
|x| x \in \mathbf{R} \text { 且 } x^{2}+1=0 \mid
$$

是空集, 因为适合条件 $x^{2}+1=0$ 的实数是不存在的. 空集记作 $\varnothing$, 且规定空集 $\varnothing$ 是任何集合 $A$ 的子集,即 $\varnothing \subset A$.

## 2. 集合的运算

集合的基本运算有以下几种:并、交、差.

设 $A 、 B$ 是两个集合,由所有属于 $A$ 或者属于 $B$ 的元䋈组成的集合,称为 $A$ 与 $B$ 的并集(简称瓶), 记作 $A \cup B$, 即

$$
A \cup B=\{x \mid x \in A \text { 或 } x \in B \mid \text {; }
$$

由所有既属于 $A$ 又属于 $B$ 的元素组成的集合, 称为 $A$ 与 $B$ 的交集(简称交), 记 作 $A \cap B$, 即

$$
A \cap B=\{x \mid x \in A \text { 且 } x \in B\} ;
$$

由所有属于 $A$ 而不属于 $B$ 的元素组成的集合, 称为 $A$ 与 $B$ 的差集(简称差), 记 作 $A \backslash B$, 即

$$
A \backslash B=\{x \mid x \in A \text { 且 } x \notin B\} .
$$

有时,我们研究某个问题限定在一个大的集合 $I$. 中进行,所研究的其他集 合 $A$ 都是 $I$ 的子集.此时,我们称集合 $I$ 为全集或基本集;称 $I \backslash A$ 为 $A$ 的余集 或补集, 记作 $A^{c}$. 例如, 在实数集 $\mathbf{R}$ 中, 集合 $A=\{x \mid 0<x \leqslant 1\}$ 的余集就是

$$
A^{c}=\{x \mid x \leqslant 0 \text { 或 } x>1\} .
$$

集合的并、交、余运算满足下列法则.

设 $A 、 B 、 C$ 为任意三个集合,则有下列法则成立:

(1) 交换律 $A \cup B=B \cup A, \quad A \cap B=B \cap A$;

(2) 结合律 $(A \cup B) \cup C=A \cup(B \cup C)$,

$$
(A \cap B) \cap C=A \cap(B \cap C) \text {; }
$$

(3) 分配律 $(A \cup B) \cap C=(A \cap C) \cup(B \cap C)$,

$$
(A \cap B) \cup C=(A \cup C) \cap(B \cup C) \text {; }
$$

(4) 对偶律

$$
\begin{aligned}
& (A \cup B)^{c}=A^{c} \cap B^{c}, \\
& (A \cap B)^{c}=A^{c} \cup B^{c} .
\end{aligned}
$$

以上这些法则都可根据集合相等的定义验证. 现就对偶律的第一个等式: “两个集合的并集的余集等于它们的余集的交集”证明如下:因为

$$
\begin{aligned}
& x \in(A \cup B)^{c} \Rightarrow x \notin A \cup B \Rightarrow x \notin A \text { 且 } x \notin B \Rightarrow x \in A^{c} \text { 且 } x \in B^{c} \\
\Rightarrow & x \in A^{c} \cap B^{c},
\end{aligned}
$$

所以 $(A \cup B)^{c} \subset A^{c} \cap B^{c}$;

反之, 因为

$x \in A^{c} \cap B^{c} \Rightarrow x \in A^{c}$ 且 $x \in B^{c} \Rightarrow x \notin A$ 且 $x \notin B \Rightarrow x \notin A \cup B$ $\Rightarrow x \in(A \cup B)^{c}$,

所以 $A^{c} \cap B^{c} \subset(A \cup B)^{c}$.

于是 $\quad(A \cup B)^{c}=A^{c} \cap B^{c}$.

注以上证明中, 符号“ $\Rightarrow$ ”表示“推出”(或“蕴含”). 如果在证明的第一段 中,将符号“ $\Rightarrow$ ”改用符号“ $\Leftrightarrow$ ”(表示“等价”)，则证明的第二段可省略.

在两个集合之间还可以定义直积或笛卡儿(Descartes)乘积. 设 $A 、 B$ 是任意 两个集合,在集合 $A$ 中任意取一个元素 $x$, 在集合 $B$ 中任意取一个元素 $y$,组成 一个有序对 $(x, y)$, 把这样的有序对作为新的元素, 它们全体组成的集合称为集 合 $A$ 与集合 $B$ 的直积, 记为 $A \times B$, 即

$$
A \times B=\{(x, y) \mid x \in A \text { 且 } y \in B\} \text {. }
$$

例如, $\mathbf{R} \times \mathbf{R}=\{(x, y) \mid x \in \mathbf{R}, y \in \mathbf{R}\}$ 即为 $x O y$ 面上全体点的集合, $\mathbf{R} \times \mathbf{R}$ 常记作 $\mathbf{R}^{2}$.

## 3. 区间和邻域

区间是用得较多的一类数集. 设 $a$ 和 $b$ 都是实数,且 $a<b$. 数集

$$
\text { - }\{x \mid a<x<b\}
$$

称为开区间,记作 $(a, b)$, 即

$$
(a, b)=|x| a<x<b \mid \text {. }
$$

$a$ 和 $b$ 称为开区间 $(a, b)$ 的端点, 这里 $a \notin(a, b), b \notin(a, b)$. 数集

$$
|x| a \leqslant x \leqslant b \mid
$$

称为闭区间, 记作 $[a, b]$, 即

$$
[a, b]=\{x \mid a \leqslant x \leqslant b\} \text {. }
$$

$a$ 和 $b$ 也称为闭区间 $[a, b]$ 的端点, 这里 $a \in[a, b], b \in[a, b]$.

类似地可说明:

$$
\begin{aligned}
& {[a, b)=\{x \mid a \leqslant x<b\},} \\
& (a, b]=\{x \mid a<x \leqslant b\} .
\end{aligned}
$$

$[a, b)$ 和 $(a, b]$ 都称为半开区间.

以上这些区间都称为有限区间. 数 $b-a$ 称为这些区间的长度. 从数轴上 看, 这些有限区间是长度为有限的线段. 闭区间 $[a, b]$ 与开区间 $(a, b)$ 在数轴上 表示出来, 分别如图 1-1(a) 与 (b) 所示. 此外还有所谓无限区间. 引进记号 $+\infty$ (读作正无穷大)及 $-\infty$ (读作负无穷大), 则可类似地表示无限区间, 例如

$$
\begin{aligned}
& {[a,+\infty)=\{x \mid x \geqslant a\},} \\
& (-\infty, b)=\{x \mid x<b\} .
\end{aligned}
$$

这两个无限区间在数轴上如图 1-1(c), (d) 所示.

全体实数的集合 $\mathbf{R}$ 也可记作 $(-\infty,+\infty)$, 它也是无限区间.

以后在不需要辦明所论区间是否包含端点, 以及是有限区间还是无限区间 的场合,我们就简单地称它为“区间”,且常用 $I$ 表示.

邻域也是一个经常用到的概念. 以点 $a$ 为中心的任何开区间称为点 $a$ 的邻 域, 记作 $U(a)$.

设 $\delta$ 是任一正数,则开区间 $(a-\delta, a+\delta)$ 就是点 $a$ 的一个邻域,这个邻域 称为点 $a$ 的 $\delta$ 邻域, 记作 $U(a, \delta)$, 即

$$
U(a, \delta)=\{x \mid a-\delta<x<a+\delta\} .
$$

点 $a$ 称为这邻域的中心, $\delta$ 称为这邻域的半径 (图 1-2).

$$
U(a, \delta)=|x||x-a|<\delta\} \text {. }
$$

因为 $|x-a|$ 表示点 $x$ 与点 $a$ 间的距离, 所以 $U(a, \delta)$ 表示: 与点 $a$ 的距离小于 $\delta$ 的一切点 $x$ 的全体.

有时用到的邻域需要把邻域中心去掉.点 $a$ 的 $\delta$ 邻域去掉中心 $a$ 后, 称为点 $a$ 的去心 $\delta$ 邻域, 记作 $\stackrel{O}{U}(a, \delta)$, 即

$$
\stackrel{\circ}{U}(a, \delta)=|x| 0<|x-a|<\delta\} .
$$

这里 $0<|x-a|$ 就表示 $x \neq a$.

为了方便,有时把开区间 $(a-\delta, a)$ 称为 $a$ 的左 $\delta$ 邻域,把开区间 $(a, a+\delta)$ 称为 $a$ 的在 $\delta$ 邻域.

两个闭区间的直积表示 $x O y$ 平面上的矩形区域.例如

$$
[a, b] \times[c, d]=\{(x, y) \mid x \in[a, b], y \in[c, d]\},
$$

即为 $x O y$ 平面上的一个矩形区域,这个区域在 $x$ 轴与 $y$ 轴上的投影分别为闭区 间 $[a, b]$ 和闭区间 $[c, d]$.

## 二、映射

## 1. 映射概念

定义 设 $X 、 Y$ 是两个非空集合, 如果存在一个法则 $f$, 使得对 $X$ 中每个元 素 $x$, 按法则 $f$, 在 $Y$ 中有唯一确定的元素 $y$ 与之对应, 则称 $f$ 为从 $X$ 到 $Y$ 的胑 射, 记作

$$
f: X \rightarrow Y,
$$

其中 $y$ 称为元素 $x$ (在映射 $f$ 下) 的像, 并记作 $f(x)$, 即

$$
y=f(x) \text {, }
$$

而元素 $x$ 称为元素 $y$ (在咉射 $f$ 下)的一个原像; 集合 $X$ 称为咉射 $f$ 的定义域, 记作 $D_{f}$, 即 $D_{f}=X ; X$ 中所有元素的像所组成的集合称为映射 $f$ 的值域, 记作 $R_{f}$ 或 $f(X)$ ，即

$$
R_{f}=f(X)=\{f(x) \mid x \in X\} .
$$

从上述映射的定义中,需要注意的是:

（1）构成一个咉射必须具备以下三个要素: 集合 $X$, 即定义域 $D_{f}=X$; 集合 $Y$, 即值域的范围: $R_{f} \subset Y$; 对应法则 $f$, 使对每个 $x \in X$, 有唯一确定的 $y=$ $f(x)$ 与之对应.

(2) 对每个 $x \in X$, 元素 $x$ 的像 $y$ 是唯一的; 而对每个 $y \in R_{f}$,元素 $y$ 的原像 不一定是唯一的; 映射 $f$ 的值域 $R_{f}$ 是 $Y$ 的一个子集, 即 $R_{f} \subset Y$, 不一定 $R_{f}=Y$.

例 1 设 $f: \mathbf{R} \rightarrow \mathbf{R}$, 对每个 $x \in \mathbf{R}, f(x)=x^{2}$. 显然, $f$ 是一个映射, $f$ 的定义 域 $D_{f}=\mathbf{R}$, 值域 $R_{f}=\{y \mid y \geqslant 0\}$, 它是 $\mathbf{R}$ 的一个真子集. 对于 $R_{f}$ 中的元素 $y$, 除 $y=0$ 外,它的原像不是唯一的. 如 $y=4$ 的原像就有 $x=2$ 和 $x=-2$ 两个.

例 2 设 $X=\left\{(x, y) \mid x^{2}+y^{2}=1\right\}, Y=\{(x, 0)|| x \mid \leqslant 1\}, f: X \rightarrow Y$, 对每 个 $(x, y) \in X$, 有唯一确定的 $(x, 0) \in Y$ 与之对应. 显然 $f$ 是一个映射, $f$ 的定 义域 $D_{f}=X$, 值域 $R_{f}=Y$. 在几何上, 这个映射表示将平面上一个圆心在原点的 单位圆周上的点投影到 $x$ 轴的区间 $[-1,1]$ 上.

例 3 设 $f:\left[-\frac{\pi}{2}, \frac{\pi}{2}\right] \rightarrow[-1,1]$, 对每个 $x \in\left[-\frac{\pi}{2}, \frac{\pi}{2}\right], f(x)=\sin x$. 这 $f$ 是一个映射, 其定义域 $D_{f}=\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$,值域 $R_{f}=[-1,1]$.

设 $f$ 是从集合 $X$ 到集合 $Y$ 的映射, 若 $R_{f}=Y$, 即 $Y$ 中任一元素 $y$ 都是 $X$ 中某元案的像, 则称 $f$ 为 $X$ 到 $Y$ 上的映射或满射; 若对 $X$ 中任意两个不同元素 $x_{1} \neq x_{2}$, 它们的像 $f\left(x_{1}\right) \neq f\left(x_{2}\right)$, 则称 $f$ 为 $X$ 到 $Y$ 的单射; 若映射 $f$ 既是单 射, 又是满射, 则称 $f$ 为二一映射 (或怒射).

上面例. 1 中的映射，既非单射，又非满射; 例 2 中的映射不是单射,是满射; 例 3 中的映射, 既是单射, 又是满射, 因此是一一映射.

映射又称为㔍子. 根据集合 $X 、 Y$ 的不同情形,在不同的数学分支中,映射 又有不同的惯用名称. 例如, 从非空集 $X$ 到数集 $Y$ 的映射又称为 $X$ 上的记函, 从非空集 $X$ 到它自身的映射又称为 $X$ 上的变换, 从实数集 (或其子集) $X$ 到实 数集 $Y$ 的映射通常称为定义在 $X$ 上的函数.

## 2. 逆映射与复合映射

设 $f$ 是 $X$ 到 $Y$ 的单射; 则由定义, 对每个 $y \in R_{f}$, 有唯一的 $x \in X$, 适合 $f(x)=y$. 于是, 我们可定义一个从 $R_{f}$ 到 $X$ 的新映射 $g$, 即

$$
g: R_{f} \rightarrow X \text {, }
$$

对每个 $y \in R_{f}$, 规定 $g(y)=x$, 这 $x$ 满足 $f(x)=y$. 这个映射 $g$ 称为 $f$ 的逆映 射,记作 $f^{-1}$ ，其定义域 $D_{f^{-1}}=R_{f}$ ，值域 $R_{f}^{-1}=X$.

按上述定义, 只有单射才存在逆咉射. 所以, 在例 $1,2,3$ 中, 只有例 3 中的映 射 $f$ 才存在逆映射 $f^{-1}$, 这个 $f^{-1}$ 就是反正弦函数的主值

$$
f^{-1}(x)=\arcsin x, x \in[-1,1] \text {, }
$$

其定义域 $D_{t^{-1}}=[-1,1]$, 值域 $R_{r^{-1}}=\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$.

设有两个映射

$$
g: X \rightarrow Y_{1}, \quad f: Y_{2} \rightarrow Z,
$$

其中 $Y_{1} \subset Y_{2}$. 则由映射 $g$ 和 $f$ 可以定出一个从 $X$ 到 $Z$ 的对应法则, 它将每个 $x \in X$ 映成 $f[g(x)] \in Z$. 显然, 这个对应法则确定了一个从 $X$ 到 $Z$ 的映射, 这 个映射称为映射 $g$ 和 $f$ 构成的复合映射, 记作 $f \circ g$, 即

$$
\begin{gathered}
f^{\circ} g: X \rightarrow Z, \\
(f \circ g)(x)=f[g(x)], x \in X .
\end{gathered}
$$

由复合映射的定义可知，映射 $g$ 和 $f$ 构成复合映射的条件是: $g$ 的值域 $R_{k}$ 必须包含在 $f$ 的定义域内, 即 $R_{g} \subset D_{l}$. 否则, 不能构成复合映射. 由此可以知 道, 映射 $g$ 和 $f$ 的复合是有顺序的, $f \circ g$ 有意义并不表示 $g \circ f$ 也有意义. 即使 $f \circ g$ 与 $g \circ f$ 都有意义,复合映射 $f \circ g$ 与 $g \circ f$ 也未必相同.

例 4 设有映射 $g: \mathbf{R} \rightarrow[-1,1]$, 对每个 $x \in \mathbf{R}, g(x)=\sin x$, 映射 $f:[-1,1]$ $\rightarrow[0,1]$, 对每个 $u \in[-1,1], f(u)=\sqrt{1-u^{2}}$. 则映射 $g$ 和 $f$ 构成的复合映射 $f \circ g: \mathbf{R} \rightarrow[0,1]$, 对每个 $x \in \mathbf{R}$, 有

$$
(f \circ g)(x)=f[g(x)]=f(\sin x)=\sqrt{1-\sin ^{2} x}=|\cos x| .
$$

## 三、函数

## 1. 函数概念

定义 设数集 $D \subset \mathbf{R}$, 则称映射 $f: D \rightarrow \mathbf{R}$ 为定义在 $D$ 上的函数, 通常简记为

$$
y=f(x), x \in D,
$$

其中 $x$ 称为息恋量, $y$ 称为因变量, $D$ 称为定义域, 记作 $D_{f}$, 即 $D_{f}=D$.

函数定义中,对每个 $x \in D$, 按对应法则 $f$, 总有唯一确定的值 $y$ 与之对应, 这个值称为函数 $f$ 在 $x$ 处的函数值, 记作 $f(x)$, 即 $y=f^{\prime}(x)$. 因变量 $y$ 与自变 量 $x$ 之间的这种依赖关系,通常称为函数关系. 函数值 $f(x)$ 的全体所构成的集 合称为函数 $f$ 的值域, 记作 $R_{f}$ 或 $f(D)$, 即

$$
R_{l}=f(D)=\left\{y \mid y=f(x), x \in D^{\prime}\right\} .
$$

需要指出,按照上述定义,记号 $f$ 和 $f(x)$ 的含义是有区别的: 前者表示自变 量 $x$ 和因变量 $y$ 之间的对应法则,而后者表示与自变量 $x$ 对应的函数值. 但为 了叙述方便, 习惯上常用记号“ $f(x), x \in D ”$ 或“ $y=f(x), x \in D ”$ 来表示定义在 $D$ 上的函数, 这时应理解为由它所确定的函数 $f$.

表示函数的记号是可以任意选取的,除了常用的 $f$ 外,还可用其他的英文

$F(x), y=\varphi(x)$ 等. 有时还直接用因变量的记号来表示函数, 即把函数记作 $y=$ $y(x)$. 但在同一个问题中,讨论到几个不同的函数时, 为了表示区别,需用不同 的记号来表示它们.

函数是从实数集到实数集的映射,其值域总在 $\mathbf{R}$ 内,因此构成函数的要素 是: 定义域 $D_{f}$ 及对应法则 $f$. 如果两个函数的定义域相同, 对应法则也相同, 那 么这两个函数就是相同的,否则就是不同的.

函数的定义域通常按以下两种情形来确定:一种是对有实际背景的函数, 根 据实际背景中变量的实际意义确定.例如; 在自由落体运动中, 设物体下落的时 间为 $t$, 下落的距离为 $s$, 开始下落的时刻 $t=0$, 落地的时刻 $t=T$, 则 $s$ 与 $t$ 之间 的函数关系是

$$
s=\frac{1}{2} g t^{2}, t \in[0, T] .
$$

这个函数的定义域就是区间 $[0, T]$; 另一种是对抽象地用算式表达的函数,通常 约定这种函数的定义域是使得算式有意义的一切实数组成的集合,这种定义域 称为函数的息然定义域. 在这种约定之下，一般的用算式表达的函数可用 “ $y=$ $f(x)$ ”表达,而不必再表出 $D_{l}$. 例如, 函数 $y=\sqrt{1-x^{2}}$ 的定义域是闭区间 $[-1,1]$, 函数 $y=\frac{1}{\sqrt{1-x^{2}}}$ 的定义域是开区间 $(-1,1)$.

在函数的定义中, 对每个 $x \in D$, 对应的函数值 $y$ 总是唯一的. 如果给定一 个对应法则, 按这个法则, 对每个 $x \in D$, 总有确定的 $y$ 值与之对应, 但这个 $y$ 不 总是唯一的,那么对于这样的对应法则并不符合函数的定义, 习惯上我们称这种 法则确定了一个多值函数. 例如, 设变量 $x$ 和 $y$ 之间的对应法则由方程 $x^{2}+y^{2}$ $=r^{2}$ 给出. 显然,对每个 $x \in[-r, r]$, 由方程 $x^{2}+y^{2}=r^{2}$ 可确定出对应的 $y$ 值, 当 $x=r$ 或 $-r$ 时,对应 $y=0$ 一个值; 当 $x$ 取 $(-r, r)$ 内任一个值时,对应 的 $y$ 有两个值. 所以这方程确定了一个多值函数. 对于多值函数, 如果我们附加 一些条件,使得在附加条件之下, 按照这个法则,对每个 $x \in D$, 总有唯一确定的 实数值 $y$ 与之对应, 那么这就确定了一个函数. 我们称这样得到的函数为多值 函数的单值分支. 例如, 在由方程 $x^{2}+y^{2}=r^{2}$ 给出的对应法则中, 附加 “ $y \geqslant 0$ ” 的 条件, 即以 “ $x^{2}+y^{2}=r^{2}$ 且 $y \geqslant 0$ ” 作为对应法则, 就可得到一个单值分支 $y=$ $y_{1}(x)=\sqrt{r^{2}-x^{2}}$; 附加 “ $y \leqslant 0$ ” 的条件, 即以 “ $x^{2}+y^{2}=r^{2}$ 且 $y \leqslant 0$ ”作为对应法 则, 就可得到另一个单值分支 $y=y_{2}(x)=-\sqrt{r^{2}-x^{2}}$.

表示函数的主要方法有三种: 表格法、图形法、解析法(公式法),这在中学里大家 已经秇悉. 其中,用图形法表示函数是基于函数图形的概念, 即坐标平面上的点集

$$
\{P(x, y) \mid y=f(x), x \in D\}
$$

称为函数 $y=f(x), x \in D$. 的图形 (图 1-3). 图中的 $R_{f}$ 表示函数 $y=f(x)$ 的值域. 下面举几个函数的例子.

例 5 函数

$$
y=2
$$

的定义域 $D=(-\infty,+\infty)$, 值域 $W=\{2\}$, 它的图形是一条平行于 $x$ 轴的直 线,如图 1-4 所示.

例 6 函数

$$
y=|x|= \begin{cases}x, & x \geqslant 0, \\ -x, & x<0\end{cases}
$$

的定义域 $D=(-\infty,+\infty)$, 值域 $R_{f}=[0,+\infty)$, 它的图形如图 1-5 所示. 这函 数称为绝对值函数.

例 7 函数

$$
y=\operatorname{sgn} x= \begin{cases}1, & x>0, \\ 0, & x=0, \\ -1, & x<0\end{cases}
$$

称为符号函数, 它的定义域 $D=(-\infty,+\infty)$, 值域 $R_{f}=\{-1,0,1\}$, 它的图形如 图 1-6 所示. 对于任何实数 $x$, 下列关系成立:

$$
x=\operatorname{sgn} x \cdot|x| \text {. }
$$

$$
y=[x]
$$

的定义域 $D=(-\infty,+\infty)$, 值域 $R_{f}=\mathbf{Z}$. 它的图形如图 1-7 所示, 这图形称为 阶梯曲线. 在 $x$ 为整数值处, 图形发生跳跃, 跃度为 1 . 这函数称为焣整函数.

在例 6 和例 7 中看到,有时一个函数要用几个式子表示. 这种在自变量的不 同变化范围中,对应法则用不同式子来表示的函数,通常称为分段函数.

例 9 函数

$$
y=f(x)= \begin{cases}2 \sqrt{x}, & 0 \leqslant x \leqslant 1, \\ 1+x, & x>1\end{cases}
$$

是一个分段函数. 它的定义域 $D=[0,+\infty)$. 当 $x \in[0,1]$ 时, 对应的函数值 $f(x)=2 \sqrt{x}$; 当 $x \in$ $(1,+\infty)$ 时, 对应的函数值 $f(x)=1+x$. 例如, $\frac{1}{2}$ $\in[0,1]$, 所以 $f\left(\frac{1}{2}\right)=2 \sqrt{\frac{1}{2}}=\sqrt{2} ; 1 \in[0,1]$, 所 以 $f(1)=2 \sqrt{1}=2 ; 3 \in(1,+\infty)$, 所以 $f(3)=1+3=4$. 这函数的图形如图 1-8 所示.

用几个式子来表示一个(不是几个!) 函数, 不仅与函数定义并无矛盾, 而且 有现实意义. 在自然科学和工程技术中, 经常会遇到分段函数的情形. 例如在等 温过程中,气体压强 $p$ 与体积 $V$ 的函数关系, 当 $V$ 不太小时依从玻意耳(Boyle)定 律; 当 $V$ 相当小时,函数关系就要用范德瓦耳斯 (van der Waals)方程来表示, 即

$$
p=\left\{\begin{array}{ll}
\frac{k}{V}, & V \geqslant V_{0}, \\
\frac{\gamma}{V-\beta}-\frac{\alpha}{V^{2}}, & \beta<V<V_{0}
\end{array} .\right.
$$

其中 $k 、 \alpha 、 \beta 、 \gamma$ 都是常量.

## 2. 函数的几种特性

(1) 函数的有界性 设函数 $f(x)$ 的定义域为 $D$, 数集 $X \subset D$. 如果存在数 $K_{1}$, 使得

$$
f(x) \leqslant K_{1}
$$

$X$ 上的一个上界. 如果存在数 $K_{2}$, 使得

$$
f(x) \geqslant K_{2}
$$

对任一 $x \in X$ 都成立, 则称函数 $f(x)$ 在 $X$ 上有下界, 而 $K_{2}$ 称为函数 $f(x)$ 在 $X$ 上的一个下界. 如果存在正数 $M$, 使得

$$
|f(x)| \leqslant M
$$

对任一 $x \in X$ 都成立, 则称函数 $f(x)$ 在 $X$ 上有界. 如果这样的 $M$ 不存在, 就称 函数 $f(x)$ 在 $X$ 上无界; 这就是说, 如果对于任何正数 $M$, 总存在 $x_{1} \in X$, 使 $\left|f\left(x_{1}\right)\right|>M$, 那么函数 $f(x)$ 在 $X$ 上无界.

例如, 就函数 $f(x)=\sin x$ 在 $(-\infty,+\infty)$ 内来说, 数 1 是它的一个上界, 数 -1 是它的一个下界(当然, 大于 1 的任何数也是它的上界, 小于 -1 的任何数也 是它的下界). 又

$$
|\sin x| \leqslant 1
$$

对任一实数 $x$ 都成立, 故函数 $f(x)=\sin x$ 在 $(-\infty,+\infty)$ 内是有界的. 这里 $M=1$ (当然也可取大于 1 的任何数作为 $M$ 而使 $|f(x)| \leqslant M$ 对任一实数 $x$ 都 成立).

又如函数 $f(x)=\frac{1}{x}$ 在开区间 $(0,1)$ 内没有上界,但有下界,例如 1 就是它的 一个下界. 函数 $f^{\prime}(x)=\frac{1}{x}$ 在开区间 $(0,1)$ 内是无界的, 因为不存在这样的正数 $M$, 使 $\left|\frac{1}{x}\right| \leqslant M$ 对于 $(0,1)$ 内的一切 $x$ 都成立 $(x$ 接近于 0 时, 不存在确定的正 数 $K_{1}$, 使 $\frac{1}{x} \leqslant K_{1}$ 成立). 但是 $f(x)=\frac{1}{x}$ 在区间 $(1,2)$ 内是有界的, 例如可取 $M$ $=1$ 而使 $\left|\frac{1}{x}\right| \leqslant 1$ 对于一切 $x \in(1,2)$ 都成立.

容易证明, 函数 $f(x)$ 在 $X$ 上有界的充分必要条件是它在 $X$ 上既有上界又有下界。

(2) 函数的单调性 设函数 $f(x)$ 的定义域为 $D$, 区间 $I \subset D$. 如果对于区间 $I$ 上任意两点 $x_{1}$ 及 $x_{2}$, 当 $x_{1}<x_{2}$ 时, 恒有

$$
f\left(x_{1}\right)<f\left(x_{2}\right),
$$

则称函数 $f(x)$ 在区间 $I$ 上是单调增加的 (图 1-9); 如果对于区间 $I$ 上任意两 点 $x_{1}$ 及 $x_{2}$, 当 $x_{1}<x_{2}$ 时, 恒有

$$
f\left(x_{1}\right)>f\left(x_{2}\right),
$$

则称函数 $f(x)$ 在区间 $I$ 上是单调减少的(图 1-10). 单调增加和单调减少的函 数统称为单调函数.

例如，函数 $f(x)=x^{2}$ 在区间 $[0,+\infty)$ 上是单调增加的，在区间 $(-\infty, 0]$ 上 是单调减少的; 在区间 $(-\infty,+\infty)$ 内函数 $f(x)=x^{2}$ 不是单调的 (图 1-11).

又例如，函数 $f(x)=x^{3}$ 在区间 $(-\infty,+\infty)$ 内是单调增加的 $($ 图 1-12).

$$
f(-x)=f(x)
$$

恒成立,则称 $f(x)$ 为偶函数. 如果对于任一 $x \in D$,

$$
f(-x)=-f(x)
$$

恒成立,则称 $f(x)$ 为奇函数.

例如, $f(x)=x^{2}$ 是偶函数, 因为 $f(-x)=(-x)^{2}=x^{2}=f(x)$. 又例如, $f(x)=x^{3}$ 是奇函数, 因为 $f(-x)=(-x)^{3}=-x^{3}=-f(x)$.

偶函数的图形关于 $y$ 轴是对称的. 因为若 $f(x)$ 是偶函数, 则 $f(-x)=$ $f(x)$, 所以如果 $A(x, f(x))$ 是图形上的点, 则与它关于 $y$ 轴对称的点 $A^{\prime}(-x$, $f(x))$ 也在图形上(图 1-13).

奇函数的图形关于原点是对称的. 因为若 $f(x)$ 是奇函数, 则 $f(-x)=$ $-f(x)$, 所以如果 $A(x, f(x))$ 是图形上的点, 则与它关于原点对称的点 $A^{\prime \prime}(-x,-f(x))$ 也在图形上(图 1-14).

函数 $y=\sin x$ 是奇函数. 函数 $y=\cos x$ 是偶函数. 函数 $y=\sin x+\cos x$ 既 非奇函数,也非偶函数.

(4) 函数的周期性 设函数 $f(x)$ 的定义域为 $D$. 如果存在一个正数 $l$, 使 得对于任一 $x \in D$ 有 $(x \pm l) \in D$, 且

$$
f(x+l)=f(x)
$$

恒成立, 则称 $f(x)$ 为周期函数, $l$ 称为 $f(x)$ 的周期, 通常我们说周期函数的周 期是指最小正周期.

例如,函数 $\sin x, \cos x$ 都是以 $2 \pi$ 为周期的周期函数; 函数 $\tan x$ 是以 $\pi$ 为 周期的周期函数.

并非每个周期函数都有最小正周期. 下面的函数就属于这种情形.

例 10 狄利克雷 (Dirichlet) 函数

$$
D(x)= \begin{cases}1, & x \in \mathbf{Q}, \\ 0, & x \in \mathbf{Q}^{c} .\end{cases}
$$

容易验证这是一个周期函数,任何正有理数 $r$ 都是它的周期. 因为不存在 最小的正有理数,所以它没有最小正周期.

## 3. 反函数与复合函数

作为逆映射的特例, 我们有以下反函数的概念.

设函数 $f: D \rightarrow f(D)$ 是单射, 则它存在逆映射 $f^{-1}: f(D) \rightarrow D$, 称此映射 $f^{-1}$ 为函数 $f$ 的反函数.

按此定义, 对每个 $y \in f(D)$, 有唯一的 $x \in D$, 使得 $f(x)=y$, 于是有

$$
f^{-1}(y)=x \text {. }
$$

这就是说，反函数 $f^{-1}$ 的对应法则是完全由函数 $f$ 的对应法则所确定的.

例如, 函数 $y=x^{3}, x \in \mathbf{R}$ 是单射, 所以它的反函数存在, 其反函数为 $x=y^{\frac{1}{3}}, y \in \mathbf{R}$.

由于习惯上自变量用 $x$ 表示, 因变量用 $y$ 表示, 于是 $y=x^{3}, x \in \mathbf{R}$ 的反函 数通常写作 $y=x^{\frac{1}{3}}, x \in \mathbf{R}$.

一般的, $y=f(x), x \in D$ 的反函数记成 $y=f^{-1}(x), x \in f(D)$.

若 $f$ 是定义在 $D$ 上的单调函数, 则 $f: D \rightarrow f(D)$ 是单射, 于是 $f$ 的反函数 $f^{-1}$ 必定存在,而且容易证明 $f^{-1}$ 也是 $f(D)$ 上的单调函数. 事实上, 不妨设 $f$ 在 $D$ 上单调增加, 现在来证明 $f^{-1}$ 在 $f(D)$ 上也是单调增加的.

任取 $y_{1}, y_{2} \in f(D)$ ，且 $y_{1}<y_{2}$. 按函数 $f$ 的定义，对 $y_{1}$, 在 $D$ 内存在唯一 的原像 $x_{1}$, 使得 $f\left(x_{1}\right)=y_{1}$, 于是 $f^{-1}\left(y_{1}\right)=x_{1}$; 对 $y_{2}$, 在 $D$ 内存在唯一的原像 $x_{2}$, 使得 $f\left(x_{2}\right)=y_{2}$, 于是 $f^{-1}\left(y_{2}\right)=x_{2}$.

如果 $x_{1}>x_{2}$, 则由 $f(x)$ 单调增加, 必有 $y_{1}>y_{2}$; 如果 $x_{1}=x_{2}$, 则显然有 $y_{1}=y_{2}$. 这两种情形都与假设 $y_{1}<y_{2}$ 不符, 故必有 $x_{1}<x_{2}$, 即 $f^{-1}\left(y_{1}\right)<$ $f^{-1}\left(y_{2}\right)$. 这就证明了 $f^{-1}$ 在 $f(D)$.上是单调增加的.

相对于反函数 $y=f^{-1}(x)$ 来说, 原来的函数 $y=f(x)$ 称为直接函数. 把直 接函数 $y=f(x)$ 和它的反函数 $y=f^{\prime}(x)$ 的图形画在同一坐标平面上, 这两个 图形关于直线 $y=x$ 是对称的 (图 1-16). 这是因为如果 $P(a, b)$ 是 $y=f(x)$ 图 形上的点, 则有 $b=f(a)$. 按反函数的定义, 有 $a=f^{-1}(b)$, 故 $Q(b, a)$ 是 $y=f^{-1}(x)$ 图形上的点; 反之, 若 $Q(b, a)$ 是 $y=f^{-1}(x)$ 图形上的点, 则 $P(a, b)$ 是 $y=f(x)$ 图形上的点. 而 $P(a, b)$ 与 $Q(b, a)$ 是关于直线 $y=x$ 对称的.

复合函数是复合映射的一种特例, 按照通常函数的记号, 复合函数的概念可 如下表述.

设函数 $y=f(u)$ 的定义域为 $D_{l}$, 函数 $u=g(x)$ 的定义域为 $D_{k}$, 且其值域 $R_{k} \subset D_{1}$, 则由下式确定的函数

$$
y=f[g(x)], \quad x \in D_{k}
$$

称为由函数 $u=g(x)$ 与函数 $y=f(u)$ 构成的复合函数, 它的定义域为 $D_{k}$, 变量 $u$ 称为中间变量.

函数 $g$ 与函数 $f$ 构成的复合函数, 即按“先 $g$ 后 $f$ ”的次序复合的函数,通常 记为 $f^{\circ} \mathrm{g}$, 即

$$
(f \circ g)(x)=f[g(x)] .
$$

与复合咉射一样, $g$ 与 $f$ 能构成复合函数 $f \circ g$ 的条件是: 函数 $g$ 的值域 $R_{n}$ 必须含在函数 $f$ 的定义域 $D_{f}$ 内, 即 $R_{g} \subset D_{f}$. 否则, 不能构成复合函数. 例如, $y=f(u)=\arcsin u$ 的定义域为 $[-1,1], u=g(x)=\sin x$ 的定义域为 $\mathbf{R}$, 且 $g(\mathbf{R}) \subset[-1,1]$, 故 $g$ 与 $f$ 可构成复合函数.

$$
y=\arcsin \sin x, x \in \mathbf{R} \text {; }
$$

又如, $y=f(u)=\sqrt{u}$ 的定义域为 $D_{f}=[0,+\infty), u=g(x)=\tan x$ 的值域为 $R_{k}$ $=(-\infty,+\infty)$, 显然 $R_{k} \not \subset D_{l}$, 故 $g$ 与 $f$ 不能构成复合函数. 但是, 如果将函数 $g$ 限制在它的定义域的一个子集 $D=\left\{x \mid k \pi \leqslant x<\left(k+\frac{1}{2}\right) \pi, k \in \mathbf{Z}\right\}$ 上, 令 $g^{*}(x)=\tan x, x \in D$, 那么 $R_{k^{*}}=g^{*}(D) \subset D_{l}, g^{*}$ 与 $f$ 就可以构成复合函数

$$
\left(f^{\circ} g^{*}\right)(x)=\sqrt{\tan x}, x \in D .
$$

习惯上为了简便起见, 仍称函数 $\sqrt{\tan x}$ 是由函数 $u=\tan x$ 与函数 $y=\sqrt{u}$ 构成的复合函数. 这里函数 $u=\tan x$ 应理解成: $u=\tan x, x \in D$. 以后, 我们采 取这种习㥂说法. 例如, 我们称函数 $u=x+1$ 与函数 $y=\ln u$ 构成复合函数 $\ln (x+1)$, 它的定义域不是 $u=x+1$ 的自然定义域 $\mathbf{R}$, 而是 $\mathbf{R}$ 的一个子集 $D=(-1,+\infty)$.

有时, 也会遇到两个以上函数所构成的复合函数, 只要它们顺次满足构成复 合函数的条件. 例如, 函数 $y=\sqrt{u}, u=\cot v, v=\frac{x}{2}$ 可构成复合函数 $y=$ $\sqrt{\cot \frac{x}{2}}$, 这里 $u$ 及 $v$ 都是中间变量, 复合函数的定义域是 $D=\{x \mid 2 k \pi<x<$ $(2 k+1) \pi, k \in \mathbf{Z}\}$, 而不是 $v=\frac{x}{2}$ 的自然定义域 $\mathbf{R}, D$ 是 $\mathbf{R}$ 的一个非空子集.

## 4. 函数的运算

设函数 $f(x), g(x)$ 的定义域依次为 $D_{1}, D_{2}, D=D_{1} \cap D_{2} \neq \varnothing$, 则我们可 以定义这两个函数的下列运算:

和(差) $f \pm g: \quad(f \pm g)(x)=f(x) \pm g(x), x \in D$;

积 $f \cdot g: \quad(f \cdot g)(x)=f(x) \cdot g(x), x \in D$;

商 $\frac{f}{g}: \quad\left(\frac{f}{g}\right)(x)=\frac{f(x)}{g(x)}, x \in D \backslash\{x \mid g(x)=0, x \in D\}$.

例 11 设函数 $f(x)$ 的定义域为 $(-l, l)$, 证明必存在 $(-l, l)$ 上的偶函数 $g(x)$ 及奇函数 $h(x)$, 使得

$$
f(x)=g(x)+h(x) .
$$

证 先分析如下: 假若这样的 $g(x) 、 h(x)$ 存在, 使得

且

$$
\begin{gathered}
f(x)=g(x)+h(x), \\
g(-x)=g(x), h(-x)=-h(x) .
\end{gathered}
$$

于是有

$$
f(-x)=g(-x)+h(-x)=g(x)-h(x) .
$$

利用 (1)、(2)式, 就可作出 $g(x) 、 h(x)$ ，这就启发我们作如下证明:

作

$$
g(x)=\frac{1}{2}[f(x)+f(-x)],
$$

$$
h(x)=\frac{1}{2}[f(x)-f(-x)] .
$$

则

$$
\begin{aligned}
& g(x)+h(x)=f(x), \\
& g(-x)=\frac{1}{2}[f(-x)+f(x)]=g(x), \\
& h(-x)=\frac{1}{2}[f(-x)-f(x)]=-h(x) .
\end{aligned}
$$

证毕.

## 5. 初等函数

在初等数学中已经讲过下面几类函数：

㧹函数: $y=x^{\prime \prime}(\mu \in \mathbf{R}$ 是常数 $)$,

指数函数: $y=a^{x}(a>0$ 且 $a \neq 1)$,

三角函数: 如 $y=\sin x, y=\cos x, y=\tan x$ 等,

反三角函数: 如 $y=\arcsin x, y=\arccos x, y=\arctan x$ 等.

以上这五类函数统称为基本初等函数.

由常数和基本初等函数经过有限次的四则运算和有限次的函数复合步骤所 构成并可用一个式子表示的函数,称为初等函数.例如

$$
y=\sqrt{1-x^{2}}, \quad y=\sin ^{2} x, \quad y=\sqrt{\cot \frac{x}{2}}
$$

等都是初等函数.在本课程中所讨论的函数绝大多数都是初等函数.

应用上常遇到以 $\mathrm{e}$ 为底的指数函数 $y=\mathrm{e}^{r}$ 和 $y=\mathrm{e}^{-x}$ 所产生的双曲函数以及 它们的反函数一反双曲函数. 它们的定义如下:

双曲正弦 $\operatorname{sh} x=\frac{\mathrm{e}^{x}-\mathrm{e}^{x}}{2}$,

双曲余弦 $\operatorname{ch} x=\frac{\mathrm{e}^{x}+\mathrm{e}^{-x}}{2}$,

叒曲正切 th $x=\frac{\operatorname{sh} x}{\operatorname{ch} x}=\frac{\mathrm{e}^{x}-\mathrm{e}^{\prime}}{\mathrm{e}^{x}+\mathrm{e}^{\prime}}$.

这三个双曲函数的简单性态如下:

“双曲正弦的定义域为 $(-\infty,+\infty)$; 它是奇函数, 它的图形通过原点且关于 原点对称，在区间 $(-\infty,+\infty)$ 内它是单调增加的. 当 $x$ 的绝对值很大时，它的

图形在第一象限内接近于曲线 $y=\frac{1}{2} \mathrm{e}^{x}$; 在第三象限内接近于曲线 $y=-\frac{1}{2} \mathrm{e}^{-x}$ (图 1-17).

双曲余弦的定义域为 $(-\infty,+\infty)$; 它是偶函数, 它的图形通过点 $(0,1)$ 且关 于 $y$ 轴对称。在区间 $(-\infty, 0)$ 内它是单调减少的; 在区间 $(0,+\infty)$ 内它是单调 增加的. ch $0=1$ 是这函数的最小值. 当 $x$ 的绝对值很大时, 它的图形在第一象 限内接近于曲线 $y=\frac{1}{2} \mathrm{e}^{r}$, 在第二象限内接近于曲线 $y=\frac{1}{2} \mathrm{e}^{-x}($ 图 1-17).

双曲正切的定义域为 $(-\infty,+\infty)$; 它是奇函 数, 它的图形通过原点且关于原点对称. 在区间 $(-\infty,+\infty)$ 内它是单调增加的. 它的图形夹在水平 直线 $y=1$ 及 $y=-1$ 之间; 且当 $x$ 的绝对值很大时, 它的图形在第一象限内接近于直线 $y=1$, 而在第三

根据双曲函数的定义, 可证下列四个公式:

$$
\begin{aligned}
& \operatorname{sh}(x+y)=\operatorname{sh} x \operatorname{ch} y+\operatorname{ch} x \operatorname{sh} y ; \\
& \operatorname{sh}(x-y)=\operatorname{sh} x \operatorname{ch} y-\operatorname{ch} x \operatorname{sh} y ; \\
& \operatorname{ch}(x+y)=\operatorname{ch} x \operatorname{ch} y+\operatorname{sh} x \operatorname{sh} y ; \\
& \operatorname{ch}(x-y)=\operatorname{ch} x \operatorname{ch} y-\operatorname{sh} x \operatorname{sh} y .
\end{aligned}
$$

我们来证明公式 (1), 其他三个公式读者可自行证明. 由定义, 得

$$
\begin{aligned}
\operatorname{sh} x \cdot \operatorname{ch} y+\operatorname{ch} x \cdot \operatorname{sh} y & =\frac{\mathrm{e}^{x}-\mathrm{e}^{-x}}{2} \cdot \frac{\mathrm{e}^{y}+\mathrm{e}^{-y}}{2}+\frac{\mathrm{e}^{x}+\mathrm{e}^{-x}}{2} \cdot \frac{\mathrm{e}^{y}-\mathrm{e}^{-y}}{2} \\
& =\frac{\mathrm{e}^{x+y}-\mathrm{e}^{y-x}+\mathrm{e}^{x-y}-\mathrm{e}^{-(x+y)}}{4}+ \\
& \frac{\mathrm{e}^{x+y}+\mathrm{e}^{y-x}-\mathrm{e}^{x-y}-\mathrm{e}^{-(x+y)}}{4} \\
& =\frac{\mathrm{e}^{x+y}-\mathrm{e}^{-(x+y)}}{2}=\operatorname{sh}(x+y) .
\end{aligned}
$$

由以上几个公式可以导出其他一些公式,例如:

在公式(4) 中令 $x=y$, 并注意到 $\operatorname{ch} 0=1$, 得

$$
\operatorname{ch}^{2} x-\operatorname{sh}^{2} x=1 ;
$$

在公式(1)中令 $x=y$, 得

$$
\operatorname{sh} 2 x=2 \operatorname{sh} x \operatorname{ch} x ;
$$

在公式(3)中令 $x=y$, 得

$$
\operatorname{ch} 2 x=\operatorname{ch}^{2} x+\operatorname{sh}^{2} x \text {. }
$$

以上关于双曲函数的公式 (1) 至 (7) 与三角函数的有关公式相类似, 把它们 对比一下可帮助记忆.

友曲函数 $y=\operatorname{sh} x, y=\operatorname{ch} x(x \geqslant 0), y=\operatorname{th} x$ 的反函数依次记为

反双曲正弦 $y=\operatorname{arsh} x$,

反双曲余弦 $y=\operatorname{arch} x$,

反茾正切 $y=\operatorname{arth} x$.

这些反双曲函数都可通过自然对数函数来表示,分别讨论如下:

先讨论双曲正弦 $y=\operatorname{sh} x$ 的反函数. 由 $x=\operatorname{sh} y$, 有

$$
x=\frac{\mathrm{e}^{y}-\mathrm{e}^{-y}}{2}
$$

令 $u=\mathrm{e}^{y}$, 则由上式有

$$
u^{2}-2 x u-1=0 .
$$

这是关于u的一个二次方程, 它的根为

$$
u=x \pm \sqrt{x^{2}+1} \text {. }
$$

因 $u=\mathrm{e}^{v}>0$, 故上式根号前应取正号, 于是

$$
u=x+\sqrt{x^{2}+1} \text {. }
$$

由于 $y=\ln u$, 故得反双曲正弦

$$
y=\operatorname{arsh} x=\ln \left(x+\sqrt{x^{2}+1}\right) .
$$

函数 $y=\operatorname{arsh} x$ 的定义域为 $(-\infty,+\infty)$, 它是奇函数, 在区间 $(-\infty,+\infty)$ 内为单调增加. 由 $y=\operatorname{sh} x$ 的图形，根据反函数的作图法, 可得 $y=\operatorname{arsh} x$ 的图 形如图 1-19 所示.

下面讨论双曲余弦 $y=\operatorname{ch} x(x \geqslant 0)$ 的反函数. 由 $x=\operatorname{ch} y(y \geqslant 0)$, 有

$$
x=\frac{\mathrm{e}^{y}+\mathrm{e}^{-y}}{2}, \quad y \geqslant 0 \text {. }
$$

由此得 $\mathrm{e}^{y}=x \pm \sqrt{x^{2}-1}$, 故

$$
y=\ln \left(x \pm \sqrt{x^{2}-1}\right) .
$$

上式中 $x$ 的值必须满足条件 $x \geqslant 1$, 而其中平方根前的符号由于 $y \geqslant 0$ 应取正. 故

$$
y=\ln \left(x+\sqrt{x^{2}-1}\right) \text {. }
$$

上述双曲余弦 $y=\operatorname{ch} x(x \geqslant 0)$ 的反函数称为反双曲余弦的主值, 记作 $y=$ $\operatorname{arch} x$, 即

$$
y=\operatorname{arch} x=\ln \left(x+\sqrt{x^{2}-1}\right) .
$$

这样规定的函数 $y=\operatorname{arch} x$ 的定义域为 $[1,+\infty)$, 它在区间 $[1,+\infty)$ 上是 单调增加的(图 $1-20$ ).

类似地可得反双曲正切

$$
y=\operatorname{arth} x=\frac{1}{2} \ln \frac{1+x}{1-x} .
$$

这函数的定义域为开区间 $(-1,1)$, 它在开区间 $(-1,1)$ 内是单调增加的奇函数. 它的图形关于原点对称 (图 1-21).

1-21

## 习 题 1-1

1. 设 $A=(-\infty,-5) \cup(5,+\infty), B=[-10,3)$, 写出 $A \cup B, A \cap B$, $A \backslash B$ 及 $A \backslash(A \backslash B)$ 的表达式.
2. 设 $A 、 B$ 是任意两个集合，证明对觔律: $(A \cap B)^{c}=A^{c} \cup B^{c}$.
3. 设映射 $f: X \rightarrow Y, A \subset X, B \subset X$. 证明:

(1) $f(A \cup B)=f(A) \cup f(B)$;

(2) $f(A \cap B) \subset f(A) \cap f(B)$.

4. 求下列函数的自然定义域:

(1) $y=\sqrt{3 x+2}$;

(2) $y=\frac{1}{1-x^{2}}$;

(3) $y=\frac{1}{x}-\sqrt{1-x^{2}}$;

(4) $y=\frac{1}{\sqrt{4-x^{2}}}$;

(5) $y=\sin \sqrt{x}$;

(6) $y=\tan (x+1)$;

(7) $y=\arcsin (x-3)$;

(8) $y=\sqrt{3-x}+\arctan \frac{1}{x}$;

(9) $y=\ln (x+1)$;

(10) $y=\mathrm{e}^{\frac{1}{4}}$.

5. 下列各题中，函数 $f(x)$ 和 $g(x)$ 是否相同? 为什么?

(1) $f(x)=\lg x^{2}, g(x)=2 \lg x$;

(2) $f(x)=x, g(x)=\sqrt{x^{2}}$;

(3) $f(x)=\sqrt[3]{x^{4}-x^{3}}, g(x)=x \sqrt[3]{x-1}$;

(4) $f(x)=1, g(x)=\sec ^{2} x-\tan ^{2} x$.

6. 设

$$
\varphi(x)=\left\{\begin{array}{cc}
|\sin x|, & |x|<\frac{\pi}{3}, \\
0, & |x| \geqslant \frac{\pi}{3},
\end{array}\right.
$$

求 $\varphi\left(\frac{\pi}{6}\right), \varphi\left(\frac{\pi}{4}\right), \varphi\left(-\frac{\pi}{4}\right), \varphi(-2)$, 并作出函数 $y=\varphi(x)$ 的图形.

7. 试证下列函数在指定区间内的单调性:
(1) $y=\frac{x}{1-x}, \quad(-\infty, 1)$;
(2) $y=x+\ln x,(0,+\infty)$.
8. 设 $f(x)$ 为定义在 $(-1,1)$ 内的奇函数, 若 $f(x)$ 在 $(0,1)$ 内塥调增加，证明 $f(x)$ 在 $(-1,0)$ 内也单调增加.
9. 设下面所考虑的函数都是定义在区间 $(-l, l)$ 上的. 证明:

（1）两个偑函数的和是偶函数,两个奇函数的和是奇函数;

是奇函数. 10. 下列函数中哪些是偶函数，哪些是奇函数、哪些既非偶函数又非奇函数?
(1) $y=x^{2}\left(1-x^{2}\right)$;
(2) $y=3 x^{2}-x^{3}$;
(3) $y=\frac{1-x^{2}}{1+x^{2}}$;
(4) $y=x(x-1)(x+1)$;
(5) $y=\sin x-\cos x+1$;
(6) $y=\frac{a^{\prime \prime}+a^{-x}}{2}$.

11. 下列各函数中哪些是周期函数? 对于周期函数，指出其周期：
(1) $y=\cos (x-2)$;
(2) $y=\cos 4 x$;
(3) $y=1+\sin \pi x$;
(4) $y=x \cos x$;
(5) $y=\sin ^{2} x$.
12. 求下列函数的反函数：
(1) $y=\sqrt[3]{x+1}$;
(2) $y=\frac{1-x}{1+x}$;
(3) $y=\frac{a x+b}{c x+d}(a d-b \neq \neq 0)$;
(4) $y=2 \sin 3 x\left(-\frac{\pi}{6} \leqslant x \leqslant \frac{\pi}{6}\right)$;
(5) $y=1+\ln (x+2)$;
(6) $y=\frac{2^{\prime}}{2^{r}+1}$.
13. 设函数 $f(x)$ 在数集 $X$ 上有定义, 试证: 函数 $f(x)$ 在 $X$ 上有界的充分必要条件是它 在 $X$ 上既有上界又有下界。
14. 在下列各题中，求由所给函数构成的复合函数，并求这函数分别对应于给定自变些 值 $x_{1}$ 和 $x_{2}$ 的函数值:

(1) $y=u^{2}, u=\sin x, x_{1}=\frac{\pi}{6}, x_{2}=\frac{\pi}{3}$;

(2) $y=\sin u, u=2, x, x_{1}=\frac{\pi}{8}, x_{2}=\frac{\pi}{4}$;

(3) $y=\sqrt{u}, u=1+x^{2}, x_{1}=1, x_{2}=2$;

(4) $y=\mathrm{e}^{u}, u=x^{2}, x_{1}=0, x_{2}=1$;

(5) $y=u^{2}, u=\mathrm{e}^{\prime}, x_{1}=1, x_{2}=-1$.

15. 设 $f(x)$ 的定义域 $D=[0,1]$, 求下列各函数的定义域:
(1) $f\left(x^{2}\right)$;
(2) $f(\sin x)$;
(3) $f(x+a)(a>0)$;
(4) $f(x+a)+f(x-a)(a>0)$.
16. 设

$$
f(x)=\left\{\begin{array}{cl}
1, & |x|<1, \\
0, & |x|=1, \quad \operatorname{rg}(x)=\mathrm{e}^{x} . \\
-1, & |x|>1,
\end{array}\right.
$$

求 $f[g(x)]$ 和 $g[f(x)]$, 并作出这两个函数的图形.

17. 已知水渠的横断面为等腰梯形,斜角 $\varphi=40^{\circ}$ (图 1-22). 当过水断面 $A B C D$ 的面积 为定值 $S_{0}$ 时, 求湿周 $L(L=A B+B C+C D)$ 与水深 $h$ 之间的函数关系式, 并指明其定 义域.

些超过 100 台以上的,每多订购 1 台,售价就降低 1 分,但最低价为每台 75 元.

(1) 将每台的实际售价 $p$ 表示为订购证 $x$ 的函数;

(2) 将厂方所获的利润 $P$ 表示成订购垃 $x$ 的函数;

（3）某一销售唡订购了1000 台,厂方可获利润多少?

19. 求联系华氏温度 (用 $F$ 表示) 和摄氏温度 (用 $C$ 表示) 的转换公式,并求

(1) $90^{\circ} \mathrm{F}$ 的等价报氏温度和 $-5{ }^{\circ} \mathrm{C}$ 的等价华氏温度;

（2）是否存在一个温度值，使华氏温度计和摄氏温度计的读数是一样的? 如果存在,那 么该温度值是多少?

20. 利用以下联合国统计办公室提供的世界人口数据以及指数模型来推测 2010 年的世 界人口。

| 年份 | 人口数(百万) | 当年人口数与上一年人口数的比值 |
| :---: | :---: | :---: |
| 1986 | 4936 |  |
| 1987 | 5023 | 1.0176 |
| 1988 | 5111 | 1.0175 |
| 1989 | 5201 | 1.0176 |
| 1990 | 5329 | 1.0246 |
| 1991 | 5422 | 1.0175 |

## 第二节 数列的极限

## 一、数列极限的定义

极限慨念是由于求某些实际问题的精确解答而产生的. 例如, 我国古代数学 家刘徽 (公元 3 世纪)利用圆内接正多边形来推算圆面积的方法一一割圆术, 就 是极限思想在几何学上的应用. 设有一圆,首先作内接正六边形, 把它的面积记为 $A_{1}$; 再作内接正十二边 形, 其面积记为 $A_{2}$; 再作内接正二十四边形, 其面积记为 $A_{3}$; 循此下去, 每次边 数加倍,一般的把内接正 $6 \times 2^{n-1}$ 边形的面积记为 $A_{n}\left(n \in \mathrm{N}^{\prime}\right)$. 这样, 就得到一 系列内接正多边形的面积:

$$
A_{1}, A_{2}, A_{3}, \cdots, A_{n}, \cdots,
$$

它们构成一列有次序的数. 当 $n$ 越大, 内接正多边形与圆的差别就越小, 从而以 $A_{n}$ 作为圆面积的近似值也越精确. 但是无论 $n$ 取得如何大, 只要 $n$ 取定了, $A_{n}$ 终究只是多边形的面积,而还不是圆的面积. 因此, 设想 $n$ 无限增大 (记为 $n \rightarrow \infty$, 读作 $n$ 趋于无穷大), 即内接正多边形的边数无限增加, 在这个过程中, 内接正 多边形无限接近于圆, 同时 $A_{n}$ 也无限接近于某一确定的数值, 这个确定的数值 就理解为圆的面积. 这个确定的数值在数学上称为上面这列有次序的数(所谓数 烈 ) $A_{1}, A_{2}, A_{3}, \cdots, A_{n}, \cdots$ 当 $n \rightarrow \infty$ 时的极限. 在圆面积问题中我们看到, 正是这 个数列的极限才精确地表达了团的面积.

在解决实际问题中逐渐形成的这种极限方法,已成为高等数学中的一种基 本方法,因此有必要作进一步的阐明.

先说明数列的概念. 如果按照某一法则, 对每个 $n \in \mathbf{N}^{+}$, 对应着一个确定的 实数 $x_{n}$, 这些实数 $x_{n}$ 按照下标 $n$ 从小到大排列得到的一个序列

就叫做数列，简记为数列 $\left\{x_{n}\right\}$.

数列中的每一个数叫做数列的项, 第 $n$ 项. $x_{n}$ 叫做数列的一般项. 例如:

$$
\begin{aligned}
& \frac{1}{2}, \frac{2}{3}, \frac{3}{4}, \cdots, \frac{n}{n+1}, \cdots ; \\
& 2,4,8, \cdots, 2^{n}, \cdots ; \\
& \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \cdots, \frac{1}{2^{n}}, \cdots ; \\
& 1,-1,1, \cdots,(-1)^{n+1}, \cdots ; \\
& 2, \frac{1}{2}, \frac{4}{3}, \cdots, \frac{n+(-1)^{n}}{n}, \cdots
\end{aligned}
$$

都是数列的例子、它们的一般项依次为

$$
\frac{n}{n+1}, \quad 2^{\prime \prime}, \frac{1}{2^{n}}, \quad(-1)^{n+1}, \frac{n+(-1)^{n-1}}{n} .
$$

在儿们上,数列 $\left\{x_{n}\right\}$ 可看作数轴上的一个动点, 它依次取数轴上的点 $x_{1}, x_{2}, x_{3}, \cdots, x_{n}, \cdots$ (图 1-23).

数列 $\left\{x_{n}\right.$ 何看作自变量为正整数 $n$ 的拯数：

$$
x_{n}=f(n), n \in \mathrm{N}^{\prime} \text {. }
$$

当自变量 $n$ 依次取 $1,2,3, \cdots$ 一切正整数时, 对应的函数值就排列成数列 $\left\{x_{n}\right\}$.

对于我们要讨论的问题来说, 重要的是: 当 $n$ 无限增大时 (即 $n \rightarrow \infty$ 时), 对 应的 $x_{n}=f(n)$ 是否能无限接近于某个确定的数值? 如果能够的话, 这个数值 等于多少?

我们对数列

$$
2, \frac{1}{2}, \frac{4}{3}, \cdots, \frac{n+(-1)^{n-1}}{n}, \cdots
$$

进行分析. 在这数列中，

$$
x_{n}=\frac{n+(-1)^{n-1}}{n}=1+(-1)^{n-1} \frac{1}{n} \text {. }
$$

我们知道,两个数 $a$ 与 $b$ 之间的接近程度可以用这两个数之差的绝对值 $|b-a|$ 来度量(在数轴上 $|b-a|$ 表示点 $a$ 与点 $b$ 之间的距离), $|b-a|$ 越小, $a$ 与b就越接近.

就数列 (1) 来说, 因为

$$
\left|x_{n}-1\right|=\left|(-1)^{n-1} \frac{1}{n}\right|=\frac{1}{n},
$$

由此可见, 当 $n$ 越来越大时, $\frac{1}{n}$ 越来越小, 从而 $x_{n}$ 就越来越接近于 1 . 因为只要 $n$ 足够大, $\left|x_{n}-1\right|$ 即 $\frac{1}{n}$ 可以小于任意给定的正数, 所以说, 当 $n$ 无限增大时, $x_{n}$ 无限接近于 1 . 例如, 给定 $\frac{1}{100}$, 欲使 $\frac{1}{n}<\frac{1}{100}$, 只要 $n>100$, 即从第 101 项起, 都能使不等式

$$
\left|x_{n}-1\right|<\frac{1}{100}
$$

成立. 同样地, 如果给定 $\frac{1}{10000}$, 则从第 10001 项起, 都能使不等式

$$
\left|x_{n}-1\right|<\frac{1}{10000}
$$

成立.一般的,不论给定的正数 $\varepsilon$ 多么小, 总存在着一个正整数 $N$, 使得当 $n>N$ 时,不等式

$$
\left|x_{n}-1\right|<\varepsilon
$$

都成立. 这就是数列 $x_{n}=\frac{n+(-1)^{n-1}}{n}(n=1,2, \cdots)$ 当 $n \rightarrow \infty$ 时无限接近于 1 这件事的实质. 这样的一个数 1 , 叫做数列 $x_{n}=\frac{n+(-1)^{n-1}}{n}(n=1,2, \cdots)$ 当 $n \rightarrow \infty$ 时的极限.

一般的,有如下数列极限的定义.

定义 设 $\left\{x_{n}\right\}$ 为一数列,如果存在常数 $a$,对于任意给定的正数 $\varepsilon$ (不论它 多么小),总存在正整数 $N$, 使得当 $n>N$ 时, 不等式

$$
\left|x_{n}-a\right|<\varepsilon
$$

都成立, 那么就称常数 $a$ 是数列 $\left\{x_{n}\right\}$ 的极限, 或者称数列 $\left\{x_{n} \mid\right.$ 收敛于 $a$, 记为

$$
\lim _{n \rightarrow \infty} x_{n}=a \text {, }
$$

或

$$
x_{n} \rightarrow a(n \rightarrow \infty) .
$$

如果不存在这样的常数 $a$, 就说数列 $\left\{x_{n}\right\}$ 没有极限, 或者说数列 $\left\{x_{n}\right\}$ 是发散的, 习惯上也说 $\lim _{n \rightarrow \infty} x_{n}$ 不存

上面定义中正数 $\varepsilon$ 可以任意给定是很重要的,因为只有这样,不等式 $\left|x_{n}-a\right|$ $<\varepsilon$ 才能表达出 $x_{n}$ 与 $a$ 无限接近的意思. 此外还应注意到: 定义中的正整数 $N$ 是与任意给定的正数 $\varepsilon$ 有关的, 它随着 $\varepsilon$ 的给定而选定.

我们给“数列 $\left\{x_{n}\right\}$ 的极限为 $a$ "一个几何解释：

将常数 $a$ 及数列 $x_{1}, x_{2}, x_{3}, \cdots, x_{n}, \cdots$ 在数轴上用它们的对应点表示出来， 再在数轴上作点 $a$ 的 $\varepsilon^{\prime}$ 邻域即开区间 $(a-\varepsilon, a+\varepsilon)$ (图 1-24).

因不等式

$$
\begin{gathered}
\left|x_{n}-a\right|<\varepsilon \\
a-\varepsilon<x_{n}<a+\varepsilon
\end{gathered}
$$

与不等式

等价,所以当 $n>N$ 时, 所有的点 $x_{n}$ 都落在开区间 $(a-\varepsilon, a+\varepsilon)$ 内, 而只有有 限个(至多只有 $N$ 个) 在这区间以外.

为了表达方便，引人记号“ $\forall$ ”表示“对于任意给定的”或“对于每一个”,记号 “ヨ”表示“存在”.于是，“对于任意给定的 $\epsilon>0$ ”正成“ $\forall \varepsilon>0$ ”, “存在正整数 $N$ ” 写成“ $\exists$ 正整数 $N$ ”, 数列极限 $\lim _{n \rightarrow \infty} x_{n}=a$ 的定义可表达为:

$$
\lim _{n \rightarrow \infty} x_{n}=a \Leftrightarrow \forall \varepsilon>0, \exists \text { 正整数 } N \text {, 当 } n>N \text { 时, 有 }\left|x_{n}-a\right|<\varepsilon \text {. }
$$

数列极限的定义㧞末直接提供如何去求数列的极限，以后要证极限的求法， 而现在只先举几个说明极限概念的例子.

例 1 证明数列

$$
2, \frac{1}{2}, \frac{4}{3}, \frac{3}{4}, \cdots, \frac{n+(-1)^{n-1}}{n}, \cdots
$$

的极限是 1 .

证 $\left|x_{n}-a\right|=\left|\frac{n+(-1)^{n-1}}{n}-1\right|=\frac{1}{n}$,

为了使 $\left|x_{n}-a\right|$ 小于任意给定的正数 $\varepsilon$ (设 $\varepsilon<1$ ), 只要

$$
\frac{1}{n}<\varepsilon \text { 或 } n>\frac{1}{\varepsilon} \text {. }
$$

所以, $\forall \varepsilon>0$, 取 $N=\left[\frac{1}{\varepsilon}\right]$, 则当 $n>N$ 时, 就有

$$
\left|\frac{n+(-1)^{n-1}}{n}-1\right|<\varepsilon,
$$

即

$$
\lim _{n \rightarrow \infty} \frac{n+(-1)^{n-1}}{n}=1 \text {. }
$$

例 2 已知 $x_{n}=\frac{(-1)^{\prime \prime}}{(n+1)^{2}}$, 证明数列 $\left\{x_{n}\right\}$ 的极限是 0 .

证 $\left|x_{n}-a\right|=\left|\frac{(-1)^{n}}{(n+1)^{2}}-0\right|=\frac{1}{(n+1)^{2}}<\frac{1}{n+1}$.

$\forall \varepsilon>0$ (设 $\varepsilon<1$ ), 只要

$$
\frac{1}{u+1}<\varepsilon \text { 或 } n>\frac{1}{\varepsilon}-1 \text { ， }
$$

不等式 $\left|x_{n}-a\right|<\varepsilon$ 必定成立. 所以, 取 $N=\left[\frac{1}{\varepsilon}-1\right]$, 则当 $n>N$ 时就有

$$
\left|\frac{(-1)^{n}}{(n+1)^{2}}-0\right|<\varepsilon \text {, }
$$

即

$$
\lim _{n \rightarrow \infty} \frac{(-1)^{n}}{(n+1)^{2}}=0 \text {. }
$$

注意 在利用数列极限的定义来论证某个数 $a$ 是数列 $\left\{x_{n}\right\}$ 的极限时, 重要 的是对于任意给定的正数 $\varepsilon$, 要能够指出定义中所说的这种正整数 $N$ 确实存 在, 但没有必要去求最小的 $N$. 如果知道 $\left|x_{n}-a\right|$ 小于某个量 (这个量是 $n$ 的一 个函数), 那么当这个量小于 $\varepsilon$ 时, $\left|x_{n}-a\right|<\varepsilon$ 当然也成立. 若令这个量小于 $\varepsilon$ 来定出 $N$ 比较方便的话,就可采用这利方法.例 2 健是这样做的.

例 3 设 $|q|<1$, 证明等比数列

的极限是 0 .

$$
1, q, q^{2}, \cdots, q^{u-1}, \cdots
$$

证 $\forall \varepsilon>0$ (设 $\varepsilon<1$ )， .

因为

$$
\left|x_{n}-0\right|=\left|q^{n-1}-0\right|=|q|^{n-1},
$$

要使 $\left|x_{u}-0\right|<\varepsilon$, 只要

$$
|q|^{n-1}<\varepsilon .
$$

取自然对数,得 $(n-1) \ln |q|<\ln \varepsilon$. 因 $|q|<1, \ln |q|<0$, 故

$$
n>1+\frac{\ln \varepsilon}{\ln |q|} \text {. }
$$

取 $N=\left[1+\frac{\ln \varepsilon}{\ln |q|}\right]$, 则当 $n>N$ 时, 就有

$$
\left|q^{\prime-1}-0\right|<\varepsilon,
$$

即 $\lim _{n \rightarrow \infty} q^{n-1}=0$.

## 二、收敛数列的性质

下面四个定理都是有关收敛数列的性质.

## 定理 1 (极限的唯一性) 如果数列 $\left\{x_{n}\right\}$ 收敛, 那么它的极限唯一.

证 用反证法. 假设同时有 $x_{n} \rightarrow a$ 及 $x_{n} \rightarrow b$, 且 $a<b$. 取 $\varepsilon=\frac{b-a}{2}$. 因为 $\lim _{n \rightarrow \infty} x_{n}=a$,故 $\exists$ 正整数 $N_{1}$, 当 $n>N_{1}$ 时,不等式

$$
\left|x_{n}-a\right|<\frac{b-a}{2}
$$

都成立. 同理, 因为 $\lim _{n \rightarrow \infty} x_{n}=b$, 故 $\exists$ 正整数 $N_{2}$, 当 $n>N_{2}$ 时, 不等式

$$
\left|x_{n}-b\right|<\frac{b-a}{2}
$$

都成立. 取 $N=\max \left\{N_{1}, N_{2}\right\}$ (这式子表示 $N$ 是 $N_{1}$ 和 $N_{2}$ 中较大的那个数), 则 当 $n>N$ 时,(2) 式及(3) 式会同时成立. 但由 (2) 式有 $x_{n}<\frac{a+b}{2}$, 由 (3) 式有 $x_{n}>\frac{a+b}{2}$,这是不可能的. 这矛盾证明了本定理的断言.

例 4 证明数列 $x_{n}=(-1)^{\prime \prime \prime}(n=1,2, \cdots)$ 是发散的.

证 如果这数列收敛, 根据定理 1 它有唯一的极限, 设极限为 $a$, 即 $\lim _{n \rightarrow \infty} x_{n}=a$. 按数列极限的定义, 对于 $\varepsilon=\frac{1}{2}, \exists$ 正整数 $N$, 当 $n>N$ 时, $\left|x_{n}-a\right|<\frac{1}{2}$ 成立; 即当 $n>N$ 时, $x_{n}$ 都在开区间 $\left(a-\frac{1}{2}, a+\frac{1}{2}\right)$ 内. 但这是不可能的, 因为 $n \rightarrow \infty$ 时, $x_{n}$ 无休止地一再重复收得 1 和 -1 这两个数, 而这两个数不可能同吋 属于长度为 1 的开区间 $\left(a-\frac{1}{2}, a+\frac{1}{2}\right)$ 内. 因此这数列发散.

下面先介绍数列的有界性概念, 然后证明收敛数列的有界性.

对于数列 $\left\{x_{n}\right\}$, 如果存在着正数 $M$, 使得对于一切 $x_{n}$ 都满足不等式

$$
\left|x_{n}\right| \leqslant M,
$$

则称数列 $\left\{x_{n}\right\}$ 是有界的; 如果这样的正数 $M$ 不存在, 就说数列 $\left\{x_{n}\right\}$ 是无界的.

例如, 数列 $x_{n}=\frac{n}{n+1}(n=1,2, \cdots)$ 是有界的, 因为可取 $M=1$, 而使

对于一切正整数 $n$ 都成立.

$$
\left|\frac{n}{n+1}\right| \leqslant 1
$$

数列 $x_{n}=2^{n}(n=1,2, \cdots)$ 是无界的, 因为当 $n$ 无限增加时, $2^{n}$ 可超过任何 正数.

数轴上对应于有界数列的点 $x_{n}$ 都落在闭区间 $[-M, M]$ 上.

定理 2(收敛数列的有界性) 如果数列 $\left\{x_{n}\right\}$ 收敛,那么数列 $\left\{x_{n}\right\}$ 一定有界.

证 因为数列 $\left\{x_{n}\right\}$ 收敛, 设 $\lim _{n \rightarrow \infty} x_{n}=a$. 根据数列极限的定义, 对于 $\varepsilon=1$, $\exists$ 正整数 $N$, 当 $n>N$ 时, 不等式

$$
\left|x_{n}-a\right|<1
$$

都成立.于是,当 $n>N$ 时,

$$
\left|x_{n}\right|=\left|\left(x_{n}-a\right)+a\right| \leqslant\left|x_{n}-a\right|+|a|<1+|a| .
$$

取 $M=\max \left\{\left|x_{1}\right|,\left|x_{2}\right|, \cdots,\left|x_{N}\right|, 1+|a|\right\}$, 那么数列 $\left\{x_{n} \mid\right.$ 中的一切 $x_{n}$ 都满足 不等式

$$
\left|x_{n}\right| \leqslant M
$$

这就证明了数列 $\left\{x_{11}\right\}$ 是有界的.

根据上述定理, 如果数列 $\left|x_{n}\right|$ 无界, 那么数列 $\left\{x_{n}\right\}$ 一定发散. 但是, 如果数 列 $\left\{x_{n}\right\}$ 有界, 却不能断定数列 $\left\{x_{n}\right\}$ 一定收敛, 例如数列

$$
1,-1,1, \cdots,(-1)^{n+1}, \cdots
$$

有界, 但例 4 证明了这数列是发散的. 所以数列有界是数列收敛的必要条件, 但 不是充分条件.

定理 3(收敛数列的保号性) 如果 $\lim _{n \rightarrow \infty} x_{n}=a$, 且 $a>0$ （或 $\left.a<0\right)$, 那么存 在正整数 $N>0$, 当 $n>N$ 时, 都有 $x_{n}>0$ (或 $\left.x_{n}<0\right)$.

证 就 $a>0$ 的情形证明. 由数列极限的定义, 对 $\varepsilon=\frac{a}{2}>0, \exists$ 正整数 $N>0$, 当 $n>N$ 时, 有

$$
\left|x_{n}-a\right|<\frac{a}{2},
$$

从而

$$
x_{n}>a-\frac{a}{2}=\frac{a}{2}>0 .
$$

推论 如果数列 $\left\{x_{n}\right\}$ 从某项起有 $x_{n} \geqslant 0$ (或 $\left.x_{n} \leqslant 0\right)$, 且 $\lim _{n \rightarrow \infty} x_{n}=a$, 那么 $a \geqslant 0$ （或 $a \leqslant 0 ）$.

证 设数列 $\left\{x_{n}\right\}$ 从第 $N_{1}$ 项起, 即当 $n>N_{1}$ 时有 $x_{n} \geqslant 0$. 现在用反证法证 明. 若 $\lim _{n \rightarrow \infty} x_{n}=a<0$, 则由定理 3 知, $\exists$ 正整数 $N_{2}>0$, 当 $n>N_{2}$ 时, 有 $x_{n}<0$. 取 $N=\max \left\{N_{1}, N_{2}\right\}$, 当 $n>N$ 时, 按假定有 $x_{n} \geqslant 0$, 按定理 3 有 $x_{n}<0$, 这引起 矛盾. 所以必有 $a \geqslant 0$.

数列 $\left\{x_{n}\right.$ \}从某项起有 $x_{n} \leqslant 0$ 的情形, 可以类似地证明.

最后,介绍子数列的概念以及关于收敛的数列与其子数列间关系的一个定 理.

在数列 $\left\{x_{n}\right\}$ 中任意抽取无限多项并保持这些项在原数列 $\left\{x_{n}\right\}$ 中的先后次 序, 这样得到的一个数列称为原数列 $\left\{x_{n}\right\}$ 的子数列 (或子列).

设在数列 $\left\{x_{n}\right\}$ 中,第一次抽取 $x_{n_{1}}$, 第二次在 $x_{n_{1}}$ 后抽取 $x_{n_{2}}$, 第三次在 $x_{n_{2}}$ 后抽取 $x_{u_{3}}, \cdots \cdots$, 这样无休止地抽取下去, 得到一个数列

$$
x_{n_{1}}, x_{n_{2}}, \cdots, x_{n_{k}}, \cdots,
$$

这个数列 $\left\{x_{n_{k}}\right\}$ 就是数列 $\left\{x_{n}\right\}$ 的一个子数列.

注意 在子数列 $\left\{x_{n_{k}}\right\}$ 中,一般项 $x_{n_{k}}$ 是第 $k$ 项,而 $x_{n_{k}}$ 在原数列 $\left\{x_{n}\right\}$ 中却是 第 $n_{k}$ 项。显然, $n_{k} \geqslant k$.

定理 4(收敛数列与其子数列间的关系) 如果数列 $\left\{x_{n}\right.$ 收敛于 $a$, 那么它 的任一子数列也收敛,且极限也是 $a$.

证 设数列 $\left\{x_{n_{k}}\right\}$ 是数列 $\left\{x_{n}\right\}$ 的任一子数列.

由于 $\lim _{n \rightarrow \infty} x_{n}=a$, 故 $\forall \varepsilon>0, \exists$ 正整数 $N$, 当 $n>N$ 时, $\left|x_{n}-a\right|<\varepsilon$ 成立.

取 $K=N$, 则当 $k>K$ 时, $n_{k}>n_{k}=n_{N} \geqslant N$. 于是 $\left|x_{n_{k}}-a\right|<\varepsilon$. 这就证明 了 $\lim _{k \rightarrow \infty} x_{n_{k}}=a$. 证毕.

由定理 4 可知,如果数列 $\left\{x_{n}\right\}$ 有两个子数列收敛于不同的极限, 那么数列 $\left\{x_{n}\right\}$ 是发散的. 例如, 例 4 中的数列

$$
1,-1,1, \cdots,(-1)^{n+1}, \cdots
$$

的子数列 $\left\{x_{2 k-1}\right\}$ 收敛于 1 , 而子数列 $\left\{x_{2 k}\right\}$ 收敛于 -1 , 因此数列 $x_{n}=(-1)^{n+1}$ $(n=1,2, \cdots)$ 是发散的. 同时这个例子也说明,一个发散的数列也可能有收敛的 子数列.

## 习 题 $1-2$

1.下列各题中, 哪些数列收敛? 哪些数列发散? 对收敛数列, 通过观察 $\mid x_{n}$ 的变化趋 势,写出它们的极限：
(1) $x_{n}=\frac{1}{2^{n}} ;$
(2) $x_{n}=(-1)^{n} \frac{1}{n}$;
(3) $x_{n}=2+\frac{1}{n^{2}}$;
(4) $x_{n}=\frac{n-1}{n+1}$;
(5) $x_{n}=n(-1)^{n}$;
(6) $x_{n}=\frac{2^{n}-1}{3^{n}}$;
(7) $x_{n}=n-\frac{1}{n}$;
(8) $x_{n}=\left[(-1)^{n}+1\right] \frac{n+1}{n}$.

2. 设数列 $\left\{x_{n}\right\}$ 的一般项 $x_{n}=\frac{1}{n} \cos \frac{n \pi}{2}$. 问 $\lim _{n \rightarrow \infty} x_{n}=$ ? 求出 $N$, 使当 $n>N$ 时, $x_{n}$ 与其极 限之差的绝对值小于正数 $\varepsilon$. 当 $\varepsilon=0.001$ 时，求出数 $N$.

3. 根据数列极限的定义证明:
(1) $\lim _{n \rightarrow \infty} \frac{1}{n^{2}}=0$;
(2) $\lim _{n \rightarrow \infty} \frac{3 n+1}{2 n+1}=\frac{3}{2}$;
(3) $\lim _{n \rightarrow \infty} \frac{\sqrt{n^{2}+a^{2}}}{n}=1$;
(4) $\lim _{n \rightarrow \infty} 0 . \underbrace{999 \cdots 9}_{n \text { 个 }}=1$

4. 若 $\lim _{n \rightarrow \infty} u_{n}=a$, 证明 $\lim _{n \rightarrow \infty}\left|u_{n}\right|=|a|$. 并举例说明: 如果数列 $\left\{\left|x_{n}\right| \mid\right.$ 有极限，但数列 $\left\{x_{n} \mid\right.$ 未必有极限.
5. 设数列 $\left\{x_{n}\right\}$ 有界, 又 $\lim _{n \rightarrow \infty} y_{n}=0$, 证明: $\lim _{n \rightarrow \infty} x_{n} y_{n}=0$.
6. 对于数列 $\left\{x_{w}\right\}$, 若 $x_{2 k-1} \rightarrow a(k \rightarrow \infty), x_{2 k} \rightarrow a(k \rightarrow \infty)$, 证明: $x_{*} \rightarrow a(n \rightarrow \infty)$.

## 第三节 函数的极限

## 一、函数极限的定义

因为数列 $\left\{x_{n}\right\}$ 可看作自变量为 $n$ 的函数: $x_{n}=f(n), n \in \mathrm{N}^{\prime}$, 所以,数列 $\left\{x_{n}\right\}$ 的极限为 $a$, 就是: 当自变量 $n$ 取正整数而无限增大 (即 $n \rightarrow \infty$ ) 时, 对应的 函数值 $f(n)$ 无限接近于确定的数 $a$. 把数列极限概念中的函数为 $f(n)$ 而自变 量的变化过程为 $n \rightarrow \infty$ 等特殊性撤开, 这样可以引出函数极限的一般概念: 在自 变量的某个变化过程中, 如果对应的函数值无限接近于某个确定的数, 那么这个 确定的数就叫做在这一变化过程中函数的极限. 这个极限是与自变量的变化过 程密切相关的,由于自变量的变化过程不同,函数的极限就表现为不同的形式. 数列极限看作函数 $f(n)$ 当 $n \rightarrow \infty$ 时的极限, 这里自变量的变化过程是 $n \rightarrow \infty$. 下面讲述自变量的变化过程为其他情形时函数 $f(x)$ 的极限, 主要研究两种情 形: （1）自变量 $x$ 任意地接近于有限值 $x_{11}$ 或者说趋于有限值 $x_{10}$ (记作 $x \rightarrow x_{01}$ ） 时,对应的函数值 $f(x)$ 的变化情形;

（2）自变量 $x$ 的绝対值 $|x|$ 无限增大即趋于无穷大 (记作 $x \rightarrow \infty$ ) 时, 对应的 函数值 $f(x)$ 的变化情形.

## 1. 自变量趋于有限值时函数的极限

现在考虑自变量 $x$ 的变化过程为 $x \rightarrow x_{0}$. 如果在 $x \rightarrow x_{11}$ 的过程中,对应的 函数值 $f(x)$ 无限接近于确定的数值 $A$, 那么就说 $A$ 是函数 $f(x)$ 当 $x \rightarrow x_{11}$ 时 的极限. 当然, 这里我们首先假定函数 $f(x)$ 在点 $x_{0}$ 的某个去心邻域内是有定义 的.

在 $x \rightarrow x_{0}$ 的过程中, 对应的函数值 $f(x)$ 无限接近于 $A$, 就是 $|f(x)-A|$ 能任意小. 如数列极限概念所述, $|f(x)-A|$ 能任意小这件事可以用 $|f(x)-A|$ $<\varepsilon$ 来表达,其中 $\varepsilon$ 是任意给定的正数. 因为函数值 $f(x)$ 无限接近于 $A$ 是在 $x \rightarrow x_{11}$ 的过程中实现的,所以对于任意给定的正数 $\varepsilon$, 只要求充分接近于 $x_{0}$ 的 $x$ 所对应的函数值 $f(x)$ 满足不等式 $|f(x)-A|<\varepsilon$; 而充分接近于 $x_{0}$ 的 $x$ 可 表达为 $0<\left|x-x_{i}\right|<\delta$, 其中 $\delta$ 是某个正数. 从几何上看,适合不等式 $0<\left|x-x_{0}\right|$ $<\delta$ 的 $x$ 的全体, 就是点 $x_{11}$ 的去心 $\delta$ 邻域, 而邻域半径 $\delta$ 则体现了 $x$ 接近 $x_{11}$ 的 程度。

通过以上分析,我们给出 $x \rightarrow x_{10}$ 时函数的极限的定义如下.

定义 1 .设函数 $f(x)$ 在点 $x_{0}$ 的某一去心邻域内有定义. 如果存在常数 $A$, 对于任意给定的正数 $\varepsilon$ (不论它多么小), 总存在正数 $\delta$, 使得当 $x$ 满足不等式 $0<\left|x-x_{11}\right|<\delta$ 时, 对应的函数值 $f(x)$ 都满足不等式

$$
|f(x)-A|<\varepsilon,
$$

那么常数 $A$ 就叫做函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时的极限, 记作

$$
\left.\lim _{x \rightarrow x_{0}} f(x)=A \text { 或 } f(x) \rightarrow A \text { (当 } x \rightarrow x_{0}\right) \text {. }
$$

我们指出,定义中 $0<\left|x-x_{0}\right|$ 表示 $x \neq x_{0}$, 所以 $x \rightarrow x_{0}$ 吋 $f(x)$ 有没有极 限, 与 $f(x)$ 在点 $x_{0}$ 是否有定义并无关系.

定义 1 可以简单地表述为:

$\lim _{x \rightarrow x_{0}} f^{\prime}(x)=A \Leftrightarrow \forall \varepsilon>0, \exists \delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 有 $|f(x)-A|<\varepsilon$.

函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时的极限为 $A$ 的几何解释如下: 任意给定一正数 $\varepsilon$,作 平行于 $x$ 轴的两条直线 $y=A+\varepsilon$ 和 $y=A-\varepsilon$,介于这两条直线之间是一横条 区域. 根据定义, 对于给定的 $\varepsilon$, 存在着点 $x_{\mathrm{v}}$ 的一个 $\delta$ 邻域 $\left(x_{\mathrm{n}}-\delta, x_{0}+\delta\right)$, 当 $y=f(x)$ 的图形上的点的横坐标 $x$ 在邻域 $\left(x_{0}-\delta, x_{11}+\delta\right)$ 内, 但 $x \neq x_{11}$ 时, 这 些点的纵坐标 $f(x)$ 满足不等式

$$
|f(x)-A|<\varepsilon,
$$

或

$$
A-\varepsilon<f(x)<A+\varepsilon .
$$

亦即这些点落在上面所作的横条区域内 (图 1-25).

例 1 证明 $\lim _{x \rightarrow c_{10}} c=c$, 此处 $c$ 为一常数.

证 这里 $|f(x)-A|=|c-c|=0$, 因此 $\forall \varepsilon>0$, 可任取 $\delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 能使不等式

$$
|f(x)-A|=|c-c|=0<\varepsilon
$$

成立. 所以 $\lim _{x \rightarrow x_{1}} c=c$.

例 2 证明 $\lim _{x \rightarrow r_{0}} x=x_{0}$.

证 这里 $|f(x)-A|=\left|x-x_{0}\right|$, 因此 $\forall \varepsilon>0$, 总可取 $\delta=\varepsilon$, 当 $0<\left|x-x_{11}\right|<\delta=\varepsilon$ 时, 能使不等式 $|f(x)-A|=\left|x-x_{11}\right|<\varepsilon$ 成立. 所以 $\lim _{r \rightarrow x_{11}} x=x_{0}$.

例 3 证明

证 由于

$$
\lim _{x \rightarrow 1}(2 x-1)=1
$$

$$
|f(x)-A|=|(2 x-1)-1|=2|x-1|,
$$

为了使 $|f(x)-A|<\varepsilon$, 只要

$$
|x-1|<\frac{\varepsilon}{2} \text {. }
$$

所以, $\forall \varepsilon>0$, 可取 $\delta=\frac{\varepsilon}{2}$, 则当 $x$ 适合不等式

$$
0<|x-1|<\delta
$$

时, 对应的函数值 $f(x)$ 就满足不等式

$$
|f(x)-1|=|(2 x-1)-1|<\varepsilon .
$$

从而

例 4 证明

$$
\lim _{x \rightarrow 1}(2 x-1)=1 \text {. }
$$

$$
\lim _{x \rightarrow 1} \frac{x^{2}-1}{x-1}=2
$$

证 这里, 函数在点 $x=1$ 是没有定义的, 但是函数当 $x \rightarrow 1$ 时的极限存在 或不存在与它并无关系. 事实上, $\forall \varepsilon>0$, 将不等式

$$
\left|\frac{x^{2}-1}{x-1}-2\right|<\varepsilon
$$

约去非零因子 $x-1$ 后, 就化为

$$
|x+1-2|=|x-1|<\varepsilon,
$$

因此, 只要取 $\delta=\varepsilon$, 那么当 $0<|x-1|<\delta$ 时, 就有

$$
\left|\frac{x^{2}-1}{x-1}-2\right|<\varepsilon \text {. }
$$

所以

$$
\lim _{x \rightarrow 1} \frac{x^{2}-1}{x-1}=2 \text {. }
$$

例 5 证明: 当 $x_{0}>0$ 时, $\lim _{x \rightarrow x_{0}} \sqrt{x}=\sqrt{x_{0}}$.

证 $\forall \varepsilon>0$, 因为

$$
|f(x)-A|=\left|\sqrt{x}-\sqrt{x_{0}}\right|=\left|\frac{x-x_{0}}{\sqrt{x}+\sqrt{x_{10}}}\right| \leqslant \frac{1}{\sqrt{x_{0}}}\left|x-x_{0}\right|,
$$

要使 $|f(x)-A|<\varepsilon$, 只要 $\left|x-x_{0}\right|<\sqrt{x_{0}} \varepsilon$ 且 $x \geqslant 0$, 而 $x \geqslant 0$ 可用 $\left|x-x_{0}\right| \leqslant x_{0}$ 保证,因此取 $\delta=\min \left\{x_{0}, \sqrt{x_{0}} \varepsilon\right\}$ (这式子表示, $\delta$ 是 $x_{01}$ 和 $\sqrt{x_{10}} \varepsilon$ 两个数中较小 的那个数), 则当 $x$ 适合不等式 $0<\left|x-x_{0}\right|<\delta$ 时, 对应的函数值 $\sqrt{x}$ 就满足 不等式

$$
\begin{aligned}
& \left|\sqrt{x}-\sqrt{x_{0}}\right|<\varepsilon . \\
& \lim _{x \rightarrow x_{0}} \sqrt{x}=\sqrt{x_{0}} \text {. }
\end{aligned}
$$

上述 $x \rightarrow x_{11}$ 时函数 $f(x)$ 的极限概念中, $x$ 是既从 $x_{0}$ 的左侧也从 $x_{01}$ 的右 侧趋于 $x_{0}$ 的.但有时只能或只需考虑 $x$ 仅从 $x_{0}$ 的左侧趋于 $x_{11}$ (记作 $x \rightarrow x_{0}^{-}$) 的情形 或 $x$ 仅从 $x_{11}$ 的右侧趋于 $x_{0}$ (记作 $x \rightarrow x_{0}^{+}$) 的情形. 在 $x \rightarrow x_{0}^{-}$的情形, $x$ 在 $x_{10}$ 的左侧, $x<x_{0}$. 在 $\lim _{x \rightarrow x_{0}} f(x)=A$ 的定义中, 把 $0<\left|x-x_{0}\right|<\delta$ 改为 $x_{0}-\delta<x<x_{0}$, 那么 $A$ 就叫做函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时的左极限, 记作

$$
\lim _{x \rightarrow x_{0}^{-}} f(x)=A \text { 或 } f\left(x_{0}^{-}\right)=A \text {. }
$$

类似地, 在 $\lim _{x \rightarrow x_{0}} f(x)=A$ 的定义中, 把 $0<\left|x-x_{0}\right|<\delta$ 改为 $x_{0}<x<x_{0}+\delta$, 那么 $A$ 就叫做函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时的右极限, 记作

$$
\lim _{x \rightarrow x_{1+}^{+}} f(x)=A \text { 或 } f\left(x_{11}^{+}\right)=A \text {. }
$$

左极限与右极限统称为单侧极限.

根据 $x \rightarrow x_{0}$ 时函数 $f(x)$ 的极限的定义以及左极限和右极限的定义, 容易 证明: 函数 $f(x)$ 当 $x \rightarrow x_{11}$ 时极限存在的充分必要条件是左极限及右极限各自 存在并且相等, 即

$$
f\left(x_{11}^{-}\right)=f\left(x_{11}^{\prime}\right) \text {. }
$$

因此, 即使 $f\left(x_{0}^{-}\right)$和 $f\left(x_{11}^{+}\right)$都存在, 但若不相等, 则 $\lim _{x \rightarrow x_{0}} f(x)$ 也不存在.

例 6 函数

$$
f(x)= \begin{cases}x-1, & x<0, \\ 0, & x=0, \\ x+1 & x>0 .\end{cases}
$$

当 $x \rightarrow 0$ 时 $f(x)$ 的极限不存在.

仿例 3 可证当 $x \rightarrow 0$ 时 $f(x)$ 的左极限

$$
\lim _{x \rightarrow 0^{-}} f(x)=\lim _{x \rightarrow 0^{-}}(x-1)=-1 \text {, }
$$

而右极限

$$
\lim _{x \rightarrow 0^{+}} f(x)=\lim _{x \rightarrow 0^{+}}(x+1)=1,
$$

因为左极限和右极限存在但不相等, 所以

$$
\lim _{x \rightarrow 11} f(x)
$$

不存在(图 1-26).

2. 自变量趋于无穷大时函数的极限

如果在 $x \rightarrow \infty$ 的过程中, 对应的函数值 $f(x)$ 无限 接近于确定的数值 $A$, 那么 $A$ 叫做函数 $f(x)$ 当 $x \rightarrow \infty$

定义 2 设函数 $f(x)$ 当 $|x|$ 大于某一正数时有定义. 如果存在常数 $A$,对 于任意给定的正数 $\varepsilon$ (不论它多么小), 总存在着正数 $X$, 使得当 $x$ 满足不等式 $|x|>X$ 时, 对应的函数值 $f(x)$ 都满足不等式

$$
|f(x)-A|<\varepsilon,
$$

那么常数 $A$ 就叫做函数 $f(x)$ 当 $x \rightarrow \infty$ 时的极限, 记作

$$
\lim _{x \rightarrow \infty} f(x)=A \text { 或 } f(x) \rightarrow A \text { (当 } x \rightarrow \infty \text { ). }
$$

定义 2 可简单地表达为:

$$
\lim _{x \rightarrow \infty} f(x)=A \Leftrightarrow \forall \varepsilon>0, \exists X>0 \text {, 当 }|x|>X \text { 时, 有 }|f(x)-A|<\varepsilon \text {. }
$$

如果 $x>0$ 且无限增大 (记作 $x \rightarrow+\infty$ ), 那么只要把上面定义中的 $|x|>X$ 改为 $x>X$, 就可得 $\lim _{x \rightarrow+\infty} f(x)=A$ 的定义: 同样, 如果 $x<0$ 而 $|x|$ 无限增大(记 作 $x \rightarrow-\infty)$, 那么只要把 $|x|>X$ 改为 $x<-X$, 便得 $\lim _{x \rightarrow-\infty} f(x)=A$ 的定义.

从几何上来说, $\lim _{x \rightarrow \infty} f(x)=A$ 的意义是: 作直线 $y=A-\varepsilon$ 和 $y=A+\varepsilon$, 则 总有一个正数 $X$ 存在,使得当 $x<-X$ 或 $x>X$ 时,函数 $y=f(x)$ 的图形位于 这两直线之间 (图 1-27). 这时, 直线 $y=A$ 是函数 $y=f(x)$ 的图形的水平渐近 线.

例 7 证明

$$
\begin{aligned}
& \text { 图 } 1-27 \\
& \lim _{x \rightarrow \infty} \frac{1}{x}=0 .
\end{aligned}
$$

证 $\forall \varepsilon>0$, 要证 $\exists X>0$, 当 $|x|>X$ 时,不等式

$$
\left|\frac{1}{x}-0\right|<\varepsilon
$$

成立.因这个不等式相当于

$$
\frac{1}{|x|}<\varepsilon
$$

或

$$
|x|>\frac{1}{\varepsilon} \text {. }
$$

由此可知, 如果取 $X=\frac{1}{\varepsilon}$, 那么当 $|x|>X=\frac{1}{\varepsilon}$ 时, 不等式 $\left|\frac{1}{x}-0\right|<\varepsilon$ 成立, 这 就证明了

$$
\lim _{x \rightarrow \infty} \frac{1}{x}=0
$$

直线 $y=0$ 是函数 $y=\frac{1}{x}$ 的图形的水平渐近线.

## 二、函数极限的性质

与收敛数列的性质相比较, 可得函数极限的一些相应的性质. 它们都可以根 据函数极限的定义, 运用类似于证明收敛数列性质的方法加以证明. 由于函数极 限的定义按自变量的变化过程不同有各种形式,下面仅以“ $\lim _{x \rightarrow x_{0}} f(x)$ ”这种形式 为代表给出关于函数极限性质的一些定理，并就其中的几个给出证明. 至于其他 形式的极限的性质及其证明,只要相应地做一些修改即可得出.

定理 1 (函数极限的唯一性) 如果 $\lim _{x \rightarrow r_{11}} f(x)$ 存在,那么这极限唯一.

定理 2 (函数极限的局部有界性) 如果 $\lim _{, \cdots, 1} f(x)=A$, 那么存在常数 $M>0$ 和 $\delta>0$, 使得当 $0<\left|x-x_{n}\right|<\delta$ 时, 有 $|f(x)| \leqslant M$.

证因为 $\lim _{x \rightarrow x_{0}} f(x)=A$, 所以取 $\varepsilon=1$, 则 $\exists \delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 有

$$
|f(x)-A|<1 \Rightarrow|f(x)| \leqslant|f(x)-A|+|A|<|A|+1,
$$

记 $M=|A|+1$, 则定理 2 就获得证明.

定理 3 (函数极限的局部保号性) 如果 $\lim _{x \rightarrow 1_{1}} f(x)=A$, 且 $A>0$ (或 $A<0$ ), 那么存在常数 $\delta>0$, 使得当 $0<\left|x-x_{0}\right|<\delta$ 时, 有 $f(x)>0$ (或 $f(x)<0$ ).

证 就 $A>0$ 的情形证明.

因为 $\lim _{x \rightarrow r_{1}} f(x)=A>0$, 所以, 取 $\varepsilon=\frac{A}{2}>0$, 则 $\exists \delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 有

$$
|f(x)-A|<\frac{A}{2} \Rightarrow f(x)>A-\frac{A}{2}=\frac{A}{2}>0 .
$$

类似地可以证明 $A<0$ 的情形.

从定理 3 的证明中可知, 在定理 3 的条件下, 可得下面更强的结论:

定理 3' 如果 $\lim _{x \rightarrow \infty} f(x)=A(A \neq 0)$, 那么就存在善 $x_{11}$ 的某一去心邻域 $\stackrel{O}{U}\left(x_{n}\right)$, 当 $x \in U^{\circ}\left(x_{0}\right)$ 时, 就有 $|f(x)|>\frac{|A|}{2}$.

由定理 3,易得以下推论.

推论 如果在 $x_{11}$ 的某去心邻域内 $f(x) \geqslant 0$ (或 $f(x) \leqslant 0$ ), 而且 $\lim _{x \rightarrow x_{0}} f(x)$ $=A$, 那么 $A \geqslant 0$ (或 $A \leqslant 0)$.

定理 4(函数极限与数列极限的关系) 如果极限 $\lim _{x \rightarrow r_{0}} f(x)$ 存在, $\left\{x_{n}\right\}$ 为 函数 $f(x)$ 的定义域内任一收敛于 $x_{0}$ 的数列, 且满足: $x_{n} \neq x_{0}\left(n \in \mathbf{N}^{\prime}\right)$, 那么相 应的函数值数列 $\left\{f\left(x_{n}\right)\right\}$ 必收敛, 且 $\lim _{n \rightarrow \infty} f\left(x_{n}\right)=\lim _{n \rightarrow r_{n}} f(x)$.

证设 $\lim _{x \rightarrow x_{0}} f(x)=A$, 则 $\forall \varepsilon>0, \exists \delta>0$, 当 $0<\left|x-x_{1}\right|<\delta$ 时, 有 $|f(x)-A|<\varepsilon$.

又因 $\lim _{n \rightarrow \infty} x_{n}=x_{n}$, 故对 $\delta>0, \exists N$, 当 $n>N$ 时, 有 $\left|x_{n}-x_{n}\right|<\delta$.

由假设, $x_{n} \neq x_{0}\left(n \in \mathbf{N}^{+}\right)$. 故当 $n>N$ 时, $0<\left|x_{n}-x_{0}\right|<\delta$, 从而 $\left|f\left(x_{n}\right)-A\right|<\varepsilon$. 即 $\lim _{n \rightarrow \infty} f\left(x_{n}\right)=A$.

## 习 题 1-3

1. 对图 1-28 所示的函数 $f(x)$, 求下列极限, 如极限不存在, 说明理由. (1) $\lim _{x \rightarrow-2} f(x)$;

(3) $\lim _{x \rightarrow 0} f(x)$.

(2) $\lim _{x \rightarrow-1} f(x)$;

2. 对图 1-29 所示的函数 $f(x)$,下列陈述中哪些是对的, 哪些是错的?
(1) $\lim _{. \rightarrow-11} f(x)$ 不存在;
(2) $\lim _{x \rightarrow 10} f(x)=0$;
(3) $\lim _{. \rightarrow 0} f(x)=1$;
(4) $\lim _{x \rightarrow 1} f(x)=0$;
(5) $\lim _{x \rightarrow 1} f(x)$ 不存在;
(6) 对每个 $x_{0} \in(-1,1), \lim _{x \rightarrow x_{0}} f(x)$ 存在. 的?
3. 对图 1-30 所示的函数,下列陈述中哪些是对的,哪些是错

(1) $\lim _{x \rightarrow-1^{+}} f(x)=1$;

(3) $\lim _{x \rightarrow 0} f(x)=0$;

(5) $\lim _{x \rightarrow 1^{-}} f(x)=1$;

(7) $\lim _{x \rightarrow 2^{-}} f(x)=0$;
(2) $\lim _{x \rightarrow-1^{-}} f(x)$ 不存在;

(4) $\lim _{x \rightarrow 0} f(x)=1$;

(6) $\lim _{x \rightarrow 1^{1}} f(x)=0$;

(8) $\lim _{x \rightarrow 2} f(x)=0$.

4. 求 $f(x)=\frac{x}{x}, \varphi(x)=\frac{|x|}{x}$ 当 $x \rightarrow 0$ 时的左、右极限, 并说明它们在 $x \rightarrow 0$ 时的极限是否存在.

5. 根据函数极限的定义证明:
(1) $\lim _{x \rightarrow 3}(3 x-1)=8$;
(2) $\lim _{x \rightarrow 2}(5 x+2)=12$;
(3) $\lim _{x \rightarrow-2} \frac{x^{2}-4}{x+2}=-4$;
(4) $\lim _{x \rightarrow-\frac{1}{2}} \frac{1-4 x^{2}}{2 x+1}=2$.

6. 根据函数极限的定义证明:
(1) $\lim _{x \rightarrow \infty} \frac{1+x^{3}}{2 x^{3}}=\frac{1}{2}$;
(2) $\lim _{x \rightarrow+\infty} \frac{\sin x}{\sqrt{x}}=0$.

7. 当 $x \rightarrow 2$ 时, $y=x^{2} \rightarrow 4$. 问 $\delta$ 等于多少, 使当 $|x-2|<\delta$ 时, $|y-4|<0.001$ ?
8. 当 $x \rightarrow \infty$ 时, $y=\frac{x^{2}-1}{x^{2}+3} \rightarrow 1$. 问 $X$ 等于多少, 使当 $|x|>X$ 时,$|y-1|<0.01$ ?
9. 证明函数 $f(x)=|x|$ 当 $x \rightarrow 0$ 时极限为零.
10. 证明: 若 $x \rightarrow+\infty$ 及 $x \rightarrow-\infty$ 时，函数 $f(x)$ 的极限都存在且都等于 $A$, 则 $\lim _{x \rightarrow \infty} f(x)=$ A.

11. 根据函数极限的定义证明: 函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时极限存在的充分必要条件是左极 限、右极限各自存在并且相等.

12. 试给出 $x \rightarrow \infty$ 时函数极限的局部有界性的定理,并加以证明.

## 第四节 无穷小与无穷大

## 一、无穷小

定义 1 如果函数 $f(x)$ 当 $x \rightarrow x_{0}$ (或 $x \rightarrow \infty$ ) 时的极限为零, 那么称函数 $f(x)$ 为当 $x \rightarrow x_{0}$ (或 $x \rightarrow \infty$ ) 时的无穷小.

特别地, 以零为极限的数列 $\left\{x_{n}\right\}$ 称为 $n \rightarrow \infty$ 时的无穷小.

例 1 因为 $\lim _{x \rightarrow 1}(x-1)=0$, 所以函数 $x-1$ 为当 $x \rightarrow 1$ 时的无穷小.

因为 $\lim _{x \rightarrow \infty} \frac{1}{x}=0$, 所以函数 $\frac{1}{x}$ 为当 $x \rightarrow \infty$ 时的无穷小.

注意 不要把无穷小与很小的数 (例如百万分之一) 混为一谈, 因为无穷小 是这样的函数, 在 $x \rightarrow x_{0}$ (或 $x \rightarrow \infty$ ) 的过程中, 这函数的绝对值能小于任意给定 的正数 $\varepsilon$, 而很小的数如百万分之一, 就不能小于任意给定的正数 $\varepsilon$, 例如取 $\varepsilon$ 等 于千万分之一, 则百万分之一就不能小于这个给定的 $\varepsilon$. 但零是可以作为无穷小 的唯一的常数, 因为如果 $f(x) \equiv 0$, 那么对于任意给定的 $\varepsilon>0$ 总有 $|f(x)|<\varepsilon$.

下面的定理说明无穷小与函数极限的关系.

定理 1 在自变量的同一变化过程 $x \rightarrow x_{\mathrm{i}}$ (或 $x \rightarrow \infty$ ) 中, 函数 $f(x)$ 具有极 限 $A$ 的充分必要条件是 $f(x)=A+\alpha$, 其中 $\alpha$ 是无穷小.

证 先证必要性. 设 $\lim _{x \rightarrow x_{0}} f(x)=A$, 则 $\forall \varepsilon>0, \exists \delta>0$, 使当 $0<\left|x-x_{0}\right|<$ $\delta$ 时, 有

$$
|f(x)-A|<\varepsilon .
$$

令 $\alpha=f(x)-A$, 则 $\alpha$ 是当 $x \rightarrow x_{0}$ 时的无穷小, 且

$$
f(x)=A+\alpha .
$$

这就证明了 $f(x)$ 等于它的极限 $A$ 与一个无穷小 $\alpha$ 之和.

再证充分性. 设 $f(x)=A+\alpha$, 其中 $A$ 是常数, $\alpha$ 是当 $x \rightarrow x_{0}$ 时的无穷小, 于是

$$
|f(x)-A|=|\alpha| \text {. }
$$

因 $\alpha$ 是当 $x \rightarrow x_{\mathrm{n}}$ 时的无穷小, 所以 $\forall \varepsilon>0, \exists \delta>0$, 使当 $0<\left|x-x_{0}\right|<\delta$ 时, 有 即

$$
|\alpha|<\varepsilon,
$$

这就证明了 $A$ 是 $f(x)$ 当 $x \rightarrow x_{0}$ 时的极限.

类似地可证明当 $x \rightarrow \infty$ 时的情形.

## 二、无穷大

如果当 $x \rightarrow x_{4}$ (或 $x \rightarrow \infty$ ) 时, 对应的函数值的绝对值 $|f(x)|$ 无限增大, 就 称函数 $f(x)$ 为当 $x \rightarrow x_{11}$ (或 $x \rightarrow \infty$ ) 时的无穷大. 精确地说, 就是

定义 2 设函数 $f(x)$ 在 $x_{\mathrm{il}}$ 的某一去心邻域内有定义 (或 $|x|$ 大于某一正 数时有定义). 如果对于任意给定的正数 $M$ (不论它多么大), 总存在正数 $\delta$ (或 正数 $X)$, 只要 $x$ 适合不等式 $0<\left|x-x_{0}\right|<\delta$ (或 $\left.|x|>X\right)$, 对应的函数值 $f(x)$ 总满足不等式

$$
|f(x)|>M,
$$

则称函数 $f(x)$ 为当 $x \rightarrow x_{0}$ (或 $x \rightarrow \infty$ ) 时的无穷大.

当 $x \rightarrow x_{0}$ (或 $x \rightarrow \infty$ ) 时的无穷大的函数 $f(x)$, 按函数极限定义来说, 极限 是不存在的. 但为了便于叙述函数的这一性态, 我们也说 “函数的极限是无穷 大”, 并记作

$$
\begin{aligned}
& \lim _{x \rightarrow x_{11}} f(x)=\infty \\
& \left(\text { 或 } \lim _{x \rightarrow \infty} f(x)=\infty\right) .
\end{aligned}
$$

如果在无穷大的定义中, 把 $|f(x)|>M$ 换成 $f(x)>M$ (或 $f(x)<-M$ ), 就记作

$$
\lim _{\substack{x \rightarrow x_{1} \\(x \rightarrow \infty)}} f(x)=+\infty \quad\left(\text { 或 } \lim _{\substack{x \rightarrow x_{0} \\(x \rightarrow \infty)}} f(x)=-\infty\right) .
$$

必须注意, 无穷大 $(\infty)$ 不是数, 不可与很大的 数 (如一千万、一亿等) 混为一谈.

例 2 证明 $\lim _{x \rightarrow 1} \frac{1}{x-1}=\infty$ (图 1-31).

证 设 $\forall M>0$.

$$
\text { 要使 } \quad \cdot\left|\frac{1}{x-1}\right|>M \text {, }
$$

只要

$$
|x-1|<\frac{1}{M} \text {. }
$$

所以, 取 $\delta=\frac{1}{M}$, 则只要 $x$ 适合不等式 $0<|x-1|$ $<\delta=\frac{1}{M}$, 就有

$$
\left|\frac{1}{x-1}\right|>M
$$

这就证明了 $\lim _{x \rightarrow 1} \frac{1}{x-1}=\infty$.

直线 $x=1$ 是函数 $y=\frac{1}{x-1}$ 的图形的铅直渐近线.

一般的说, 如果 $\lim _{x \rightarrow x_{0}} f(x)=\infty$, 则直线 $x=x_{0}$ 是函数 $y=f(x)$ 的图形的铅 直渐近线.

无穷大与无穷小之间有一种简单的关系, 即 :

定理 2 在自变量的同一变化过程中, 如果 $f(x)$ 为无穷大, 则 $\frac{1}{f(x)}$ 为无穷 小; 反之, 如果 $f(x)$ 为无穷小, 且 $f(x) \neq 0$, 则 $\frac{1}{f(x)}$ 为无穷大.

证 设 $\lim _{x \rightarrow+x_{0}} f(x)=\infty$.

$\forall \varepsilon>0$. 根据无穷大的定义, 对于 $M=\frac{1}{\varepsilon}, \exists \delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 有

$$
|f(x)|>M=\frac{1}{\varepsilon},
$$

即

$$
\left|\frac{1}{f^{\prime}(x)}\right|<\varepsilon \text {, }
$$

所以 $\frac{1}{f(x)}$ 为当 $x \rightarrow x_{0}$ 时的无穷小.

反之, 设 $\lim _{x \rightarrow x_{0}} f(x)=0$, 且 $f(x) \neq 0$.

$\forall M>0$. 根据无穷小的定义, 对于 $\varepsilon=\frac{1}{M}, \exists \delta>0$, 当 $0<\left|x-x_{0}\right|<\delta$ 时, 有

$$
|f(x)|<\varepsilon=\frac{1}{M},
$$

由于当 $0<\left|x-x_{0}\right|<\delta$ 时 $f(x) \neq 0$, 从而

$$
\left|\frac{1}{f(x)}\right|>M \text {, }
$$

所以 $\frac{1}{f(x)}$ 为当 $x \rightarrow x_{0}$ 时的无穷大.

类似地可证当 $x \rightarrow \infty$ 时的情形.

## 习 题 1-4

1. 两个无穷小的商是否一定是无穷小? 举例说明之.

2. 根据定义证明:
(1) $y=\frac{x^{2}-9}{x+3}$ 为当 $x \rightarrow 3$ 时的无穷小;
(2) $y=x \sin \frac{1}{x}$ 为当 $x \rightarrow 0$ 时的无穷小.

3. 根据定义证明: 函数 $y=\frac{1+2 x}{x}$ 为当 $x \rightarrow 0$ 时的无穷大. 问 $x$ 应满足什么条件,能使 $|y|>10^{4}$ ?
4. 求下列极限并说明理由:
(1) $\lim _{x \rightarrow \infty} \frac{2 x+1}{x}$;
(2) $\lim _{x \rightarrow 0} \frac{1-x^{2}}{1-x}$
5. 根据函数极限或无穷大定义,填写下表:

|  | $f(x) \rightarrow A$ | $f(x) \rightarrow \infty$ | $f(x) \rightarrow+\infty$ | $f(x) \rightarrow-\infty$ |
| :---: | :---: | :---: | :---: | :---: |
| $x \rightarrow x_{n}$ | $\begin{array}{l}\forall \varepsilon>0, \\ \exists \delta>0, \\ \text { 使当 } 0<\left\|x-x_{0}\right\|<\delta \text { 时, } \\ \text { 即有 }\|f(x)-A\|<\varepsilon .\end{array}$ |  |  |  |
| $x \rightarrow x_{\\| \prime}^{\prime}$ |  |  |  |  |
| $x \rightarrow x_{0}^{-}$ |  |  |  |  |
| $x \rightarrow \infty$ |  | $\begin{array}{l}\forall M>0, \\ \exists X>0, \\ \text { 使当 }\|x\|>X \text { 时, } \\ \text { 即有 }\|f(x)\|>M .\end{array}$ |  |  |
| $x \rightarrow+\infty$ |  |  |  |  |
| $x \rightarrow-\infty$ |  |  |  |  |

6. 函数 $y=x \cos x$ 在 $(-\infty,+\infty)$ 内是否有界? 这个函数是否为 $x \rightarrow+\infty$ 时的无穷大? 为什么?
7. 证明: 函数 $y=\frac{1}{x} \sin \frac{1}{x}$ 在区间 $(0,1]$ 上无界,但这函数不是 $x \rightarrow 0^{+}$时的无穷大.
8. 求函数 $f(x)=\frac{4}{2-x^{2}}$ 的图形的渐近线.

## 第五节 极限运算法则

本节讨论极限的求法, 主要是建立极限的四则运算法则和复合函数的极限 运算法则, 利用这些法则, 可以求某些函数的极限. 以后我们还将介绍求极限的 其他方法.

在下面的讨论中, 记号“lim”下面没有标明自变量的变化过程,实际上,下面 的定理对 $x \rightarrow x_{0}$ 及 $x \rightarrow \infty$ 都是成立的. 在论证时, 我们只证明了 $x \rightarrow x_{0}$ 的情形, 只要把 $\delta$ 改成 $X$, 把 $0<\left|x-x_{0}\right|<\delta$ 改成 $|x|>X$, 就可得 $x \rightarrow \infty$ 情形的证明.

## 定理 1 有限个无穷小的和也是无穷小.

证 考虑两个无穷小的和.

设 $\alpha$ 及 $\beta$ 是当 $x \rightarrow x_{0}$ 时的两个无穷小,而

$$
\gamma=\alpha+\beta \text {. }
$$

$\forall \varepsilon>0$. 因为 $\alpha$ 是当 $x \rightarrow x_{0}$ 时的无穷小, 对于 $\frac{\varepsilon}{2}>0, \exists \delta_{1}>0$, 当 $0<\left|x-x_{0}\right|$ $<\delta_{1}$ 时, 不等式

$$
|\alpha|<\frac{\varepsilon}{2}
$$

成立. 又因 $\beta$ 是当 $x \rightarrow x_{0}$ 时的无穷小, 对于 $\frac{\varepsilon}{2}>0, \exists \delta_{2}>0$, 当 $0<\left|x-x_{0}\right|<\delta_{2}$ 时,不等式

$$
|\beta|<\frac{\varepsilon}{2}
$$

成立. 取 $\delta=\min \left\{\delta_{1}, \delta_{2} \mid\right.$, 则当 $0<\left|x-x_{0}\right|<\delta$ 时，

$$
|\alpha|<\frac{\varepsilon}{2} \text { 及 }|\beta|<\frac{\varepsilon}{2}
$$

同时成立, 从而 $|\gamma|=|\alpha+\beta| \leqslant|\alpha|+|\beta|<\frac{\varepsilon}{2}+\frac{\varepsilon}{2}=\varepsilon$. 这就证明了 $\gamma$ 也是当 $x \rightarrow x_{0}$ 时的无穷小.

有限个无穷小之和的情形可以同样证明.

## 定理 2 有界函数与无穷小的乘积是无穷小.

证 设函数 $u$ 在 $x_{0}$ 的某一去心邻域 $\stackrel{\cup}{U}\left(x_{n}, \delta_{1}\right)$ 内是有界的, 即 $\exists M>0$ 使 $|u| \leqslant M$ 对一切 $x \in \stackrel{\circ}{U}\left(x_{v}, \delta_{1}\right)$ 成立. 又设 $\alpha$ 是当 $x \rightarrow x_{v}$ 时的无穷小, 即 $\forall \varepsilon>0$, $\exists \delta_{2}>0$, 当 $x \in \stackrel{\circ}{U}\left(x_{0}, \delta_{2}\right)$ 时, 有

$$
|\alpha|<\frac{\varepsilon}{M} .
$$

取 $\delta=\min \left\{\delta_{1}, \delta_{2}\right\}$, 则当 $x \in U^{\circ}\left(x_{0}, \delta\right)$ 时,

$$
|u| \leqslant M \quad \text { 及 }|\alpha|<\frac{\varepsilon}{M}
$$

同时成立.从而

$$
|u \alpha|=|u| \cdot|\alpha|<M \cdot \frac{\varepsilon}{M}=\varepsilon,
$$

这就证明了 $u \dot{\alpha}$ 是当 $x \rightarrow x_{0}$ 时的无穷小.

## 推论 1 常数与无穷小的乘积是无穷小.

推论 2 有限个无穷小的乘积也是无穷小.

定理 3 如果 $\lim f(x)=A, \lim g(x)=B$, 那么

(1) $\lim [f(x) \pm g(x)]=\lim f(x) \pm \lim g(x)=A \pm B$;

(2) $\lim [f(x) \cdot g(x)]=\lim f(x) \cdot \lim g(x)=A \cdot B$;

(3) 若又有 $B \neq 0$, 则

$$
\lim \frac{f(x)}{g(x)}=\frac{\lim f(x)}{\lim g(x)}=\frac{A}{B} .
$$

证 先证 (1).

因 $\lim f(x)=A, \operatorname{limg}(x)=B$, 由第四节定理 1 有

$$
f(x)=A+\alpha, \quad g(x)=B+\beta,
$$

其中 $\alpha$ 及 $\beta$ 为无穷小. 于是

$$
f(x) \pm g(x)=(A+\alpha) \pm(B+\beta)=(A \pm B)+(\alpha \pm \beta) .
$$

由本节定理 $1, \alpha \pm \beta$ 是无穷小 $(\alpha-\beta$ 可看作 $\alpha+(-1) \beta$, 由本节定理 2 的推论 $1,(-1) \beta$ 是无穷小, 因此 $\alpha-\beta$ 世可看作两个无穷小的和). 再由第四节定理 1 , 得

$$
\lim [f(x) \pm g(x)]=A \pm B=\lim f(x) \pm \lim g(x) .
$$

关于 (2) 的证明, 建议读者作为练习.

再证 (3).

由 $\lim f(x)=A, \lim g(x)=B$, 有

$$
f(x)=A+\alpha, g(x)=B+\beta,
$$

其中 $\alpha$ 及 $\beta$ 为无穷小. 设

$$
\gamma=\frac{f(x)}{g(x)}-\frac{A}{B},
$$

则

$$
\gamma=\frac{A+\alpha}{B+\beta}-\frac{A}{B}=\frac{1}{B(B+\beta)}(B \alpha-A \beta) .
$$

上式表示, $\gamma$ 可看作两个函数的乘积,其中函数 $B \alpha-A \beta$ 是无穷小.下面我们证 明另一个函数 $\frac{1}{B(B+\beta)}$ 在点 $x_{0}$ 的某一邻域内有界. 根据第三节定理 $3^{\prime}$, 由于 $\operatorname{limg}(x)=B \neq 0$, 存在着点 $x_{0}$ 的某一去心邻域 $\stackrel{O}{U}\left(x_{0}\right)$, 当 $x \in \stackrel{O}{U}\left(x_{0}\right)$ 时, $|g(x)|>\frac{|B|}{2}$, 从而 $\left|\frac{1}{g(x)}\right|<\frac{2}{|B|}$. 于是

$$
\left|\frac{1}{B(B+\beta)}\right|=\frac{1}{|B|} \cdot\left|\frac{1}{g(x)}\right|<\frac{1}{|B|} \cdot \frac{2}{|B|}=\frac{2}{|B|^{2}} \text {. }
$$

这就证明了 $\frac{1}{B(B+\beta)}$ 在点 $x_{0}$ 的去心邻域 $U^{\circ}\left(x_{11}\right)$ 内有界.

因此，根据本节定理 $2, \gamma$ 是无穷小. 而

所以由上节定理 1 , 得

$$
\frac{f(x)}{g(x)}=\frac{A}{B}+\gamma
$$

$$
\lim \frac{f(x)}{g(x)}=\frac{A}{B}=\frac{\lim f(x)}{\lim g(x)}
$$

证毕.

定理 3 中的 (1)、(2) 可推广到有限个函数的情形. 例如, 如果 $\lim f(x)$, $\lim g(x), \lim h(x)$ 都存在, 则有

$$
\begin{aligned}
& \lim [f(x)+g(x)-h(x)]=\lim f(x)+\lim g(x)-\lim h(x), \\
& \lim [f(x) \cdot g(x) \cdot h(x)]=\lim f(x) \cdot \lim g(x) \cdot \lim h(x) .
\end{aligned}
$$

关于定理 3 中的(2), 有如下推论:

推论 1 如果 $\lim f(x)$ 存在, 而 $c$ 为常数, 则

$$
\lim [c f(x)]=c \lim f(x) .
$$

就是说, 求极限时, 常数因子可以提到极限记号外面. 这是因为 $\lim c=c$.

推论 2 如果 $\lim f(x)$ 存在, 而 $n$ 是正整数, 则

$$
\lim [f(x)]^{n}=[\lim f(x)]^{n} .
$$

这是园为

$$
\begin{aligned}
\lim [f(x)]^{\prime \prime} & =\lim [f(x) \cdot f(x) \cdots f(x)] \\
& =\lim f(x) \cdot \lim f(x) \cdots \lim f(x)=[\lim f(x)]^{n} .
\end{aligned}
$$

关于数列, 也有类似的极限四则运算法则, 这就是下面的定理.

定理 4 设有数列 $\left\{x_{n}\right\}$ 和 $\left.\mid y_{n}\right\}$. 如果

$$
\lim _{n \rightarrow \infty} x_{n}=A, \quad \lim _{n \rightarrow \infty} y_{n}=B \text {, }
$$

那么

(1) $\lim _{n \rightarrow \infty}\left(x_{n} \pm y_{n}\right)=A \pm B$;

(2) $\lim _{n \rightarrow \infty} x_{n} \cdot y_{n}=A \cdot B$;

(3) 当 $y_{n} \neq 0(n=1,2, \cdots)$ 且 $B \neq 0$ 时, $\lim _{n \rightarrow \infty} \frac{x_{n}}{y_{n}}=\frac{A}{B}$.

## 证明从略.

定理 5 如果 $\varphi(x) \geqslant \psi(x)$, 而 $\lim \varphi(x)=a, \lim \psi(x)=b$, 那么 $a \geqslant b$.

证 令 $f(x)=\varphi(x)-\psi(x)$, 则 $f(x) \geqslant 0$. 由本节定理 3 有

$$
\begin{aligned}
\lim f(x) & =\lim [\varphi(x)-\psi(x)] \\
& =\lim \varphi(x)-\lim \psi(x)=a-b .
\end{aligned}
$$

由第三节定理 3 推论, 有 $\lim f(x) \geqslant 0$, 即 $a-b \geqslant 0$, 故 $a \geqslant b$.

例 1 求 $\lim _{x \rightarrow 1}(2 x-1)$.

解 $\lim _{x \rightarrow 1}(2 x-1)=\lim _{x \rightarrow 1} 2 x-\lim _{x \rightarrow 1} 1=2 \lim _{x \rightarrow 1} x-1=2 \cdot 1-1=1$.

例 2 求 $\lim _{x \rightarrow 2} \frac{x^{3}-1}{x^{2}-5 x+3}$.

解 这里分母的极限不为零, 故

$$
\begin{aligned}
\lim _{x \rightarrow 2} \frac{x^{3}-1}{x^{2}-5 x+3} & =\frac{\lim _{x \rightarrow 2}\left(x^{3}-1\right)}{\lim _{x \rightarrow 2}\left(x^{2}-5 x+3\right)} \\
& =\frac{\lim _{x \rightarrow 2} x^{3}-\lim _{x \rightarrow 2} 1}{\lim _{x \rightarrow 2} x^{2}-5 \lim _{x \rightarrow 2} x+\lim _{x \rightarrow 2} 3}=\frac{\left(\lim _{x \rightarrow 2} x\right)^{3}-1}{\left(\lim _{x \rightarrow 2} x\right)^{2}-5 \cdot 2+3} \\
& =\frac{2^{3}-1}{2^{2}-10+3}=\frac{7}{-3}=-\frac{7}{3} .
\end{aligned}
$$

从上面两个例子可以看出, 求有理整函数(多项式)或有理分式函数当 $x \rightarrow x_{0}$ 的极限时,只要把 $x_{0}$ 代替函数中的 $x$ 就行了; 但是对于有理分式函数, 这样代入后如果分母等于零,则没有意义.

事实上,设多项式

$$
f(x)=a_{0} x^{n}+a_{1} x^{n-1}+\cdots+a_{n},
$$

则 $\lim _{x \rightarrow x_{0}} f(x)=\lim _{x \rightarrow x_{11}}\left(a_{0} x^{n}+a_{1} x^{n-1}+\cdots+a_{n}\right)$

$$
\begin{aligned}
& =a_{0}\left(\lim _{x \rightarrow t_{0}} x\right)^{n}+a_{1}\left(\lim _{x \rightarrow x_{0}} x\right)^{n-1}+\cdots+\lim _{x \rightarrow x_{0}} a_{n} \\
& =a_{0} x_{n}^{n}+a_{1} x_{n}^{n-1}+\cdots+a_{n}=f\left(x_{0}\right) ;
\end{aligned}
$$

又设有理分式函数

$$
F(x)=\frac{P(x)}{Q(x)}
$$

其中 $P(x), Q(x)$ 都是多项式, 于是

$$
\lim _{x \rightarrow x_{0}} P(x)=P\left(x_{10}\right), \quad \lim _{x \rightarrow x_{10}} Q(x)=Q\left(x_{0}\right) ;
$$

如果 $Q\left(x_{0}\right) \neq 0$, 则

$$
\lim _{x \rightarrow x_{0}} F(x)=\lim _{x \rightarrow x_{0}} \frac{P(x)}{Q(x)}=\frac{\lim _{x \rightarrow x_{0}} P(x)}{\lim _{x \rightarrow x_{0}} Q(x)}=\frac{P\left(x_{0}\right)}{Q\left(x_{0}\right)}=F\left(x_{0}\right) .
$$

但必须注意: 若 $Q\left(x_{10}\right)=0$, 则关于商的极限的运算法则不能应用, 那就需 要特别考虑.下面我们举两个属于这种情形的例题.

例 3 求 $\lim _{x \rightarrow 3} \frac{x-3}{x^{2}-9}$.

解 当 $x \rightarrow 3$ 时, 分子及分母的极限都是零, 于是分子、分母不能分别取极 限. 因分子及分母有公因子 $x-3$, 而 $x \rightarrow 3$ 时, $x \neq 3, x-3 \neq 0$, 可约去这个不为 零的公因子.所以

$$
\lim _{x \rightarrow 3} \frac{x-3}{x^{2}-9}=\lim _{x \rightarrow 3} \frac{1}{x+3}=\frac{\lim _{x \rightarrow 3} 1}{\lim _{x \rightarrow 3}(x+3)}=\frac{1}{6} .
$$

例 4 求 $\lim _{x \rightarrow 1} \frac{2 x-3}{x^{2}-5 x+4}$.

解 因为分母的极限 $\lim _{x \rightarrow 1}\left(x^{2}-5 x+4\right)=1^{2}-5 \cdot 1+4=0$, 不能应用商的极 限的运算法则. 但因

$$
\lim _{x \rightarrow 1} \frac{x^{2}-5 x+4}{2 x-3}=\frac{1^{2}-5 \cdot 1+4}{2 \cdot 1-3}=0,
$$

故由第四节定理 2 得

$$
\lim _{x \rightarrow 1} \frac{2 x-3}{x^{2}-5 x+4}=\infty .
$$

例 5 求 $\lim _{x \rightarrow \infty} \frac{3 x^{3}+4 x^{2}+2}{7 x^{3}+5 x^{2}-3}$.

解 先用 $x^{3}$ 去除分母及分子, 然后取极限:

$$
\lim _{x \rightarrow \infty} \frac{3 x^{3}+4 x^{2}+2}{7 x^{3}+5 x^{2}-3}=\lim _{x \rightarrow \infty} \frac{3+\frac{4}{x}+\frac{2}{x^{3}}}{7+\frac{5}{x}-\frac{3}{x^{3}}}=\frac{3}{7},
$$

这是因为 $\quad \lim _{x \rightarrow \infty} \frac{a}{x^{n}}=a \lim _{x \rightarrow \infty} \frac{1}{x^{n}}=a\left(\lim _{x \rightarrow \infty} \frac{1}{x}\right)^{n}=0$,

其中 $a$ 为常数, $n$ 为正整数, $\lim _{x \rightarrow \infty} \frac{1}{x}=0$ (见第三节例 7).

例 6 求 $\lim _{x \rightarrow \infty} \frac{3 x^{2}-2 x-1}{2 x^{3}-x^{2}+5}$.

解 先用 $x^{3}$ 除分母:和分子, 然后求极限, 得

$$
\lim _{x \rightarrow \infty} \frac{3 x^{2}-2 x-1}{2 x^{3}-x^{2}+5}=\lim _{x \rightarrow \infty} \frac{\frac{3}{x}-\frac{2}{x^{2}}-\frac{1}{x^{3}}}{2-\frac{1}{x}+\frac{5}{x^{3}}}=\frac{0}{2}=0 .
$$

例 7 求 $\lim _{x \rightarrow \infty} \frac{2 x^{3}-x^{2}+5}{3 x^{2}-2 x-1}$. 解 应用例 6 的结果并根据上节定理 2 , 即得

$$
\lim _{x \rightarrow \infty} \frac{2 x^{3}-x^{2}+5}{3 x^{2}-2 x-1}=\infty .
$$

例 5、例 $6 、$ 例 7 是下列一般情形的特例, 即当 $a_{10} \neq 0, b_{10} \neq 0, m$ 和 $n$ 为非负 整数时, 有:

$$
\lim _{x \rightarrow \infty} \frac{a_{0} x^{m}+a_{1} x^{m-1}+\cdots+a_{m}}{b_{0} x^{n}+b_{1} x^{n-1}+\cdots+b_{n}}=\left\{\begin{array}{l}
\frac{a_{0}}{b_{0}}, \text { 当 } n=m, \\
0, \text { 当 } n>m, \\
\infty, \text { 当 } n<m .
\end{array}\right.
$$

例 8 求 $\lim _{x \rightarrow \infty} \frac{\sin x}{x}$.

解 当 $x \rightarrow \infty$ 时, 分子及分母的极限都不存在, 故关于商的极限的运算法则 不能应用. 如果把 $\frac{\sin x}{x}$ 看作 $\sin x$ 与 $\frac{1}{x}$ 的乘积, 由于 $\frac{1}{x}$ 当 $x \rightarrow \infty$ 时为无穷小, 而 $\sin x$ 是有界函数,则根据本节定理 2 , 有

$$
\lim _{x \rightarrow \infty} \frac{\sin x}{x}=0 .
$$

定理 6(复合函数的极限运算法则) 设函数 $y=f[g(x)]$ 是由函数 $u=g(x)$ 与函数 $y=f(u)$ 复合而成, $f[g(x)]$ 在点 $x_{0}$ 的某去心邻域内有定义, 若 $\lim _{x \rightarrow x_{0}} g(x)=u_{0}, \lim _{u \rightarrow u_{0}} f(u)=A$, 且存在 $\delta_{0}>0$, 当 $x \in \dot{U}^{0}\left(x_{0}, \delta_{0}\right)$ 时, 有 $g(x) \neq$ $u_{0}$, 则

$$
\lim _{x \rightarrow x_{0}} f[g(x)]=\lim _{u \rightarrow u_{0}} f(u)=A .
$$

证 按函数极限的定义, 要证: $\forall \varepsilon>0, \exists \delta>0$, 使得当 $0<\left|x-x_{0}\right|<\delta$ 时,

$$
|f[g(x)]-A|<\varepsilon
$$

成立.

由于 $\lim _{u \rightarrow u_{0}} f(u)=A, \forall \varepsilon>0, \exists \eta>0$, 当 $0<\left|u-u_{0}\right|<\eta$ 时, $|f(u)-A|<\varepsilon$ 成立.

又由于 $\lim _{x \rightarrow x_{0}} g(x)=u_{0}$, 对于上面得到的 $\eta>0, \exists \delta_{1}>0$, 当 $0<\left|x-x_{0}\right|<\delta_{1}$ 时, $\left|g(x)-u_{0}\right|<\eta$ 成立.

由假设, 当 $x \in \dot{U}^{\circ}\left(x_{0}, \delta_{0}\right)$ 时, $g(x) \neq u_{0}$. 取 $\delta=\min \left\{\delta_{0}, \delta_{1}\right\}$, 则当 $0<$ $\left|x-x_{0}\right|<\delta$ 时, $\left|g(x)-u_{0}\right|<\eta$ 及 $\left|g(x)-u_{0}\right| \neq 0$ 同时成立, 即 $0<$ $\left|g(x)-u_{n}\right|<\eta$ 成立, 从而 成立. 证毕.

$$
|f[g(x)]-A|=|f(u)-A|<\varepsilon
$$

在定理 6 中, 把 $\lim _{x \rightarrow x_{0}} g(x)=u_{0}$ 换成 $\lim _{x \rightarrow x_{0}} g(x)=\infty$ 或 $\lim _{x \rightarrow \infty} g(x)=\infty$, 而把 $\lim _{u \rightarrow u_{u}} f(u)=A$ 换成 $\lim _{n \rightarrow \infty} f(u)=A$, 可得类似的定理.

定理 6 表示, 如果函数 $g(x)$ 和 $f(u)$ 满足该定理的条件, 那么作代换 $u=$ $g(x)$ 可把求 $\lim _{x \rightarrow x_{0}} f[g(x)]$ 化为求 $\lim _{u \rightarrow u_{u}} f(u)$, 这里 $u_{u}=\lim _{x \rightarrow x_{0}} g(x)$.

## 习 题 $1-5$

1. 计算下列极限:
(1) $\lim _{x \rightarrow 2} \frac{x^{2}+5}{x-3}$;
(2) $\lim _{x \rightarrow \sqrt{3}} \frac{x^{2}-3}{x^{2}+1}$
(3) $\lim _{x \rightarrow 1} \frac{x^{2}-2 x+1}{x^{2}-1}$;
(4) $\lim _{x \rightarrow 0} \frac{4 x^{3}-2 x^{2}+x}{3 x^{2}+2 x}$;
(5) $\lim _{h \rightarrow 0} \frac{(x+h)^{2}-x^{2}}{h}$;
(6) $\lim _{x \rightarrow \infty}\left(2-\frac{1}{x}+\frac{1}{x^{2}}\right)$;
(7) $\lim _{x \rightarrow \infty} \frac{x^{2}-1}{2 x^{2}-x-1}$;
(8) $\lim _{x \rightarrow \infty} \frac{x^{2}+x}{x^{4}-3 x^{2}+1}$;
(9) $\lim _{x \rightarrow 4} \frac{x^{2}-6 x+8}{x^{2}-5 x+4}$;
(10) $\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)\left(2-\frac{1}{x^{2}}\right)$;
(11) $\lim _{n \rightarrow \infty}\left(1+\frac{1}{2}+\frac{1}{4}+\cdots+\frac{1}{2^{n}}\right)$;
(12) $\lim _{n \rightarrow \infty} \frac{1+2+3+\cdots+(n-1)}{n^{2}}$;
(13) $\lim _{n \rightarrow \infty} \frac{(n+1)(n+2)(n+3)}{5 n^{3}}$;
(14) $\lim _{1 \rightarrow 1}\left(\frac{1}{1-x}-\frac{3}{1-x^{3}}\right)$.
2. 计算下列极限:
(1) $\lim _{x \rightarrow 2} \frac{x^{3}+2 x^{2}}{(x-2)^{2}}$;
(2) $\lim _{x \rightarrow \infty} \frac{x^{2}}{2 x+1}$;
(3) $\lim _{x \rightarrow \infty}\left(2 x^{3}-x+1\right)$.
3. 计算下列极限:
(1) $\lim _{x \rightarrow 0} x^{2} \sin \frac{1}{x}$;
(2) $\lim _{x \rightarrow \infty} \frac{\arctan x}{x}$.
4. 设 $\left\{a_{n} \mid,\left\{b_{n}\right\},\left\{c_{n}\right.\right.$ 均为非负数列, 且 $\lim _{n \rightarrow \infty} a_{n}=0, \lim _{n \rightarrow \infty} b_{n}=1, \lim _{n \rightarrow \infty} c_{n}=\infty$. 下列陈述中哪 些是对的，哪些是错的? 如果是对的, 说明理由; 如果是错的, 试给出一个反例.
(1) $a_{n}<b_{n}, n \in \mathbf{N}^{+}$;
(2) $b_{n}<c_{n}, n \in \mathbf{N}^{+}$;
(3) $\lim _{n \rightarrow \infty} a_{n} c_{n}$ 不存在;
(4) $\lim _{n \rightarrow \infty} b_{n} c_{n}$ 不存在.
5. 下列陈述中, 哪些是对的、哪些是错的? 如果是对的, 说明理由; 如果是错的, 试给出 一个反例.

(1) 如果 $\lim _{x \rightarrow x_{10}} f(x)$ 存在, 但 $\lim _{x \rightarrow x_{11}} g(x)$ 不存在, 那么 $\lim _{x \rightarrow x_{0}}[f(x)+g(x)]$ 不存在; (2) 如果 $\lim _{x \rightarrow x_{0}} f(x)$ 和 $\lim _{x \rightarrow r_{0}} g(x)$ 都不存在,那么 $\lim _{x \rightarrow x_{0}}[f(x)+g(x)]$ 不存在;

（3）如果 $\lim _{x \rightarrow 1_{11}} f(x)$ 存在,但 $\lim _{x \rightarrow x_{0}} g(x)$ 不存在,那么 $\lim _{x \rightarrow x_{0}} f(x) \cdot g(x)$ 不存在.

*6. 证明本节定理 3 中的(2).

## 第六节 极限存在准则 两个重要极限

下面讲判定极限存在的两个准则以及作为应用准则的例子, 讨论两个重要 极限 $: \lim _{x \rightarrow 0} \frac{\sin x}{x}=1$ 及 $\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)^{r}=\mathrm{e}$.

准则 I 如果数列 $\left\{x_{n}\right\} 、\left\{y_{n}\right\}$ 及 $\left|z_{n}\right|$ 满足下列条件:

(1) 从某项起, 即 $\exists n_{1} \in \mathrm{N}$, 当 $n>n_{0}$ 时, 有

$$
y_{n} \leqslant x_{n} \leqslant z_{n} \text {, }
$$

(2) $\lim _{n \rightarrow \infty} y_{n}=a, \lim _{n \rightarrow \infty} z_{n}=a$,

那么数列 $\left\{x_{n}\right\}$ 的极限存在, 且 $\lim _{n \rightarrow \infty} x_{n}=a$.

证 因 $y_{n} \rightarrow a, z_{n} \rightarrow a$, 所以根据数列极限的定义, $\forall \varepsilon>0, \exists$ 正整数 $N_{1}$, 当 $n>N_{1}$ 时,有 $\left|y_{n}-a\right|<\varepsilon$; 又 $\exists$ 正整数 $N_{2}$, 当 $n>N_{2}$ 时,有 $\left|z_{n}-a\right|<\varepsilon$. 现在 取 $N=\max \left|n_{0}, N_{1}, N_{2}\right|$, 则当 $n>N$ 时, 有

$$
\left|y_{n}-a\right|<\varepsilon, \quad\left|z_{n}-a\right|<\varepsilon
$$

同时成立, 即

$$
a-\varepsilon<y_{u}<a+\varepsilon, \quad a-\varepsilon<z_{n}<a+\varepsilon
$$

同时成立. 又因当 $n>N$ 时, $x_{n}$ 介于 $y_{n}$ 和 $z_{n}$ 之间, 从而有

$$
a-\varepsilon<y_{n} \leqslant x_{n} \leqslant z_{n}<a+\varepsilon,
$$

即

$$
\left|x_{n}-a\right|<\varepsilon
$$

成立.这就证明了 $\lim _{n \rightarrow \infty} x_{n}=a$.

上述数列极限存在准则可以推广到函数的极限：

准则 I 如果

(1) 当 $x \in \stackrel{\circ}{U}\left(x_{0}, r\right)$ (或 $|x|>M$ ) 时,

$$
g(x) \leqslant f(x) \leqslant h(x)
$$

(2) $\lim _{\substack{x \rightarrow x_{0} \\(x \rightarrow \infty)}} g(x)=A, \lim _{\substack{x \rightarrow x_{11} \\(x \rightarrow \infty)}} h(x)=A$,

那么 $\lim _{\substack{x \rightarrow x_{0} \\(x \rightarrow \infty)}} f(x)$ 存在, 且等于 $A$.

准则 $\mathrm{I}$ 及准则 $\mathrm{I}^{\prime}$ 称为夹逼准则. 作为准则 $\mathrm{I}^{\prime}$ 的应用, 下面证明一个重要的极限

$$
\lim _{x \rightarrow 0} \frac{\sin x}{x}=1 \text {. }
$$

首先注意到, 函数 $\frac{\sin x}{x}$ 对于一切 $x \neq 0$ 都有定义.

在图 $1-32$ 所示的四分之一的单位圆中, 设圆心角 $\angle A O B=$ $x\left(0<x<\frac{\pi}{2}\right)$, 点 $A$ 处的切线与 $O B$ 的延长线相交 于 $D$, 又 $B C \perp O A$, 则

$$
\sin x=C B, x=\overparen{A B}, \tan x=A D \text {. }
$$

因为

$\triangle A O B$ 的面积 $<$ 扇形 $A O B$ 的面积 $<\triangle A O D$ 的面积, 所以

$$
\frac{1}{2} \sin x<\frac{1}{2} x<\frac{1}{2} \tan x,
$$

即

$$
\sin x<x<\tan x .
$$

不等号各边都除以 $\sin x$, 就有

或

$$
\begin{aligned}
& 1<\frac{x}{\sin x}<\frac{1}{\cos x}, \\
& \cos x<\frac{\sin x}{x}<1 .
\end{aligned}
$$

因为当 $x$ 用 $-x$ 代替时, $\cos x$ 与 $\frac{\sin x}{x}$ 都不变,所以上面的不等式对于开区间 $\left(-\frac{\pi}{2}, 0\right)$ 内的一切 $x$ 也是成立的.

为了对 (1) 式应用准则 $I^{\prime}$, 下面来证 $\lim _{x \rightarrow 01} \cos x=1$.

事实上,当 $0<|x|<\frac{\pi}{2}$ 时,

$$
0<|\cos x-1|=1-\cos x=2 \sin ^{2} \frac{x}{2}<2\left(\frac{x}{2}\right)^{2}=\frac{x^{2}}{2},
$$

即

$$
0<1-\cos x<\frac{x^{2}}{2}
$$

当 $x \rightarrow 0$ 时, $\frac{x^{2}}{2} \rightarrow 0$, 由准则 $I^{\prime}$ 有 $\lim _{x \rightarrow 0}(1-\cos x)=0$, 所以

$$
\lim _{x \rightarrow 0} \cos x=1 \text {. }
$$

由于 $\lim _{x \rightarrow 0} \cos x=1, \lim _{x \rightarrow 0} 1=1$, 由不等式(1)及准则 $I^{\prime}$, 即得

$$
\lim _{x \rightarrow 0} \frac{\sin x}{x}=1 \text {. }
$$

从图 1-33 中, 也可以看出这个重要极限.

例 1 求 $\lim _{x \rightarrow 11} \frac{\tan x}{x}$.

解 $\lim _{x \rightarrow 0} \frac{\tan x}{x}=\lim _{x \rightarrow 0}\left(\frac{\sin x}{x} \cdot \frac{1}{\cos x}\right)$

$$
=\lim _{x \rightarrow 0} \frac{\sin x}{x} \cdot \lim _{x \rightarrow 11} \frac{1}{\cos x}=1 \text {. }
$$

例 2 求 $\lim _{x \rightarrow 0} \frac{1-\cos x}{x^{2}}$.

解 $\lim _{x \rightarrow 0} \frac{1-\cos x}{x^{2}}=\lim _{x \rightarrow 0} \frac{2 \sin ^{2} \frac{x}{2}}{x^{2}}=\frac{1}{2} \lim _{x \rightarrow 0} \frac{\sin ^{2} \frac{x}{2}}{\left(\frac{x}{2}\right)^{2}}$

$$
=\frac{1}{2} \lim _{x \rightarrow 0}\left(\frac{\sin \frac{x}{2}}{\frac{x}{2}}\right)^{2}=\frac{1}{2} \cdot 1^{2}=\frac{1}{2} \text {. }
$$

这里倒数第二个等号用到了复合函数的极限运算法则. 实际上, $\frac{\sin \frac{x}{2}}{\frac{x}{2}}$ 可看 作由 $\frac{\sin u}{u}$ 及 $u=\frac{x}{2}$ 复合而成. 因 $\lim _{r \rightarrow 0} \frac{x}{2}=0$, 而 $\lim _{n \rightarrow 0} \frac{\sin u}{u}=1$, 故

$$
\lim _{x \rightarrow 11} \frac{\sin \frac{x}{2}}{\frac{x}{2}}=\lim _{u \rightarrow 0} \frac{\sin u}{u}=1 \text {. }
$$

例 3 求 $\lim _{x \rightarrow 10} \frac{\arcsin x}{x}$.

解 令 $t=\arcsin x$, 则 $x=\sin t$, 当 $x \rightarrow 0$ 时,有 $t \rightarrow 0$. 于是由复合函数的极 限运算法则得

$$
\lim _{x \rightarrow 01} \frac{\arcsin x}{x}=\lim _{t \rightarrow 0} \frac{t}{\sin t}=1 .
$$

准则 II 单调有界数列必有极限. 如果数列 $\left\{x_{n}\right\}$ 满足条件

$$
x_{1} \leqslant x_{2} \leqslant x_{3} \leqslant \cdots \leqslant x_{n} \leqslant x_{n+1} \leqslant \cdots,
$$

就称数列 $\left\{x_{n}\right\}$ 是单调增加的; 如果数列 $\left\{x_{n}\right\}$ 满足条件

$$
x_{1} \geqslant x_{2} \geqslant x_{3} \geqslant \cdots \geqslant x_{n} \geqslant x_{n+1} \geqslant \cdots,
$$

就称数列 $\left\{x_{n}\right\}$ 是单调减少的. 单调增加和单调减少的数列统称为单调数列 $\mathbb{1}$.

在第二节中曾证明:收敛的数列一定有界. 但那时也曾指出 : 有界的数列不 一定收敛. 现在准则 II 表明: 如果数列不仅有界, 并且是单调的, 那么这数列的极 限必定存在,也就是这数列一定收敛.

对准则 II 我们不作证明，而给出如下的几何解释.

从数轴上看,对应于单调数列的点 $x_{n}$ 只可能向一个方向移动, 所以只有两 种可能情形: 或者点 $x_{n}$ 沿数轴移向无穷远 $\left(x_{n} \rightarrow+\infty\right.$ 或 $\left.x_{n} \rightarrow-\infty\right)$; 或者点 $x_{n}$ 无限趋近于某一个定点 $A$ (图 1-34), 也就是数列 $\left\{x_{n}\right\}$ 趋于一个极限. 但现在假 定数列是有界的, 而有界数列的点 $x_{n}$ 都落在数轴上某一个区间 $[-M, M]$ 内, 那么上述第一种情形就不可能发生了. 这就表示这个数列趋于一个极限, 并且这 个极限的绝对值不超过 $M$.

作为准则 II 的应用, 我们讨论另一个重要极限

$$
\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)^{x}
$$

下面考虑 $x$ 取正整数 $n$ 而趋于 $+\infty$ 的情形.

$$
\begin{aligned}
\text { 设 } x_{n}= & \left(1+\frac{1}{n}\right)^{n}, \text { 我们来证数列 }\left\{x_{n}\right\} \text { 单调增加并且有界. 按牛顿二项公式, 有 } \\
x_{n}= & \left(1+\frac{1}{n}\right)^{n} \\
= & 1+\frac{n}{1 !} \cdot \frac{1}{n}+\frac{n(n-1)}{2 !} \cdot \frac{1}{n^{2}}+\frac{n(n-1)(n-2)}{3 !} \cdot \frac{1}{n^{3}}+\cdots+ \\
& \frac{n(n-1) \cdots(n-n+1)}{n !} \cdot \frac{1}{n^{n}} \\
= & 1+1+\frac{1}{2 !}\left(1-\frac{1}{n}\right)+\frac{1}{3 !}\left(1-\frac{1}{n}\right)\left(1-\frac{2}{n}\right)+\cdots+ \\
& \frac{1}{n !}\left(1-\frac{1}{n}\right)\left(1-\frac{2}{n}\right) \cdots\left(1-\frac{n-1}{n}\right),
\end{aligned}
$$

种广义的单调数列. 类似地,

$$
\begin{aligned}
x_{n+1}= & 1+1+\frac{1}{2 !}\left(1-\frac{1}{n+1}\right)+\frac{1}{3 !}\left(1-\frac{1}{n+1}\right)\left(1-\frac{2}{n+1}\right)+\cdots+ \\
& \frac{1}{n !}\left(1-\frac{1}{n+1}\right)\left(1-\frac{2}{n+1}\right) \cdots\left(1-\frac{n-1}{n+1}\right)+ \\
& \frac{1}{(n+1) !}\left(1-\frac{1}{n+1}\right)\left(1-\frac{2}{n+1}\right) \cdots\left(1-\frac{n}{n+1}\right) .
\end{aligned}
$$

比较 $x_{n} 、 x_{n+1}$ 的展开式, 可以看到除前两项外, $x_{n}$ 的每一项都小于 $x_{n+1}$ 的对应 项,并且 $x_{n+1}$ 还多了最后的一项, 其值大于 0 ,因此

$$
x_{n}<x_{n+1},
$$

这就说明数列 $\left\{x_{n}\right\}$ 是单调增加的. 这个数列同时还是有界的. 因为, 如果 $x_{n}$ 的 展开式中各项括号内的数用较大的数 1 代替, 得

$$
\begin{aligned}
x_{n} & <1+1+\frac{1}{2 !}+\frac{1}{3 !}+\cdots+\frac{1}{n !}<1+1+\frac{1}{2}+\frac{1}{2^{2}}+\cdots+\frac{1}{2^{n-1}} \\
& =1+\frac{1-\frac{1}{2^{n}}}{1-\frac{1}{2}}=3-\frac{1}{2^{n-1}}<3,
\end{aligned}
$$

这就说明数列 $\left\{x_{n}\right\}$ 是有界的. 根据极限存在准则 II , 这个数列 $\left\{x_{n}\right\}$ 的极限存在, 通常用字母 e 来表示它, 即

$$
\lim _{n \rightarrow \infty}\left(1+\frac{1}{n}\right)^{n}=\mathrm{e} .
$$

可以证明, 当 $x$ 取实数而趋于 $+\infty$ 或 $-\infty$ 时, 函数 $\left(1+\frac{1}{x}\right)^{r}$ 的极限都存在 且都等于 $e^{(1)}$. 因此(2),

(1) 设 $n \leqslant x<n+1$, 则

且 $n$ 与 $x$ 同时趋于 $+\infty$, 因为

应用夹逼准则，即得

$$
\begin{gathered}
\lim _{n \rightarrow \infty}\left(1+\frac{1}{n+1}\right)^{n}=\lim _{n \rightarrow \infty} \frac{\left(1+\frac{1}{n+1}\right)^{n+1}}{1+\frac{1}{n+1}}=\mathrm{e}, \\
\lim _{n \rightarrow \infty}\left(1+\frac{1}{n}\right)^{n+1}=\lim _{n \rightarrow \infty}\left[\left(1+\frac{1}{n}\right)^{n} \cdot\left(1+\frac{1}{n}\right)\right]=\mathrm{e},
\end{gathered}
$$

$$
\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)^{x}=c \text {. }
$$

令 $x=-(t+1)$, 则 $x \rightarrow-\infty$ 时, $t \rightarrow+\infty$. 从而

$$
\begin{aligned}
\lim _{x \rightarrow-\infty}\left(1+\frac{1}{x}\right)^{t} & =\lim _{t \rightarrow+\infty}\left(1-\frac{1}{t+1}\right)^{-(t+1)}=\lim _{t \rightarrow \infty}\left(\frac{t}{t+1}\right)^{-(t+1)} \\
& =\lim _{t \rightarrow+\infty}\left(1+\frac{1}{t}\right)^{t+1}=\lim _{t \rightarrow+\infty}\left[\left(1+\frac{1}{t}\right)^{\prime} \cdot\left(1+\frac{1}{t}\right)\right]=c .
\end{aligned}
$$

(2)参阅习題 1-3 第 10 题

$$
\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)^{. r}=\mathrm{e}
$$

这个数 $\mathrm{e}$ 是无理数, 它的值是

$$
\mathrm{e}=2.718281828459045 \cdots \text {. }
$$

在第一节中提到的指数函数 $y=\mathrm{e}^{\cdot r}$ 以及自然对数 $y=\ln x$ 中的底 $\mathrm{e}$ 就是这个常 数.

利用复合函数的极限运算法则, 可把 (2) 式写成另一形式. 在 $(1+z)^{\frac{1}{\varepsilon}}$ 中作 代换 $x=\frac{1}{z}$, 得 $\left(1+\frac{1}{x}\right)^{x}$. 又当 $z \rightarrow 0$ 时 $x \rightarrow \infty$. 因此由复合函数的极限运算法则 得

$$
\lim _{z \rightarrow 0}(1+z)^{\frac{1}{z}}=\lim _{x \rightarrow \infty}\left(1+\frac{1}{x}\right)^{x}=\mathrm{e} .
$$

下面的例 4 也是用代换方法来做的, 实质上还是用到了复合函数的极限运 算法则.

例 4 求 $\lim _{x \rightarrow \infty}\left(1-\frac{1}{x}\right)^{x}$.

解 令 $t=-x$, 则当 $x \rightarrow \infty$ 时, $t \rightarrow \infty$. 于是

$$
\lim _{x \rightarrow \infty}\left(1-\frac{1}{x}\right)^{x}=\lim _{t \rightarrow \infty}\left(1+\frac{1}{t}\right)^{-t}=\lim _{t \rightarrow \infty} \frac{1}{\left(1+\frac{1}{t}\right)^{t}}=\frac{1}{\mathrm{e}} .
$$

相应于单调有界数列必有极限的准则 II , 函数极限也有类似的准则. 对于自 变量的不同变化过程 $\left(x \rightarrow x_{\mathfrak{u}}^{-}, x \rightarrow x_{0}^{+}, x \rightarrow-\infty, x \rightarrow+\infty\right)$, 准则有不同的形 式. 现以 $x \rightarrow x_{0}^{-}$为例, 将相应的准则叙述如下:

准则 $\mathbb{I}^{\prime}$ 设函数 $f(x)$ 在点 $x_{0}$ 的某个左邻域内单调并且有界, 则 $f(x)$ 在 $x_{0}$ 的左极限 $f\left(x_{0}^{-}\right)$必定存在.

在第二节例 1 及例 2 中, 我们看到收敛数列不一定是单调的. 因此, 准则 II 所给出的单调有界这条件, 是数列收敛的充分条件, 而不是必要的. 当然, 其中有 界这一条件对数列的收敛性来说是必要的.下面叙述的柯西极限存在准则, 它给 出了数列收敛的充分必要条件.

柯西极限存在准则 数列 $\left\{x_{n} \mid\right.$ 收敛的充分必要条件是: 对于任意给定的正 数 $\varepsilon$, 存在着这样的正整数 $N$, 使得当 $m>N, n>N$ 时, 就有

(1) 柯西(Augustin Louis Cauchy,1789-1857)、法国数学家,1821 年他出版了《分析教程》、《无穷小

义，将微积分理论完整而严密地基基于极限的基础之上，从而使他成为严格微积分学的奠基者.

$$
\left|x_{n}-x_{u}\right|<\varepsilon .
$$

证必要性 设 $\lim _{n \rightarrow \infty} x_{n}=a . \forall \varepsilon>0$, 由数列极限的定义, $\exists$ 正整数 $N$,当 $n>N$ 时,有

$$
\left|x_{n}-a\right|<\frac{\varepsilon}{2}
$$

同样,当 $m>N$ 时,也有

$$
\left|x_{m}-a\right|<\frac{\varepsilon}{2} .
$$

因此, 当 $m>N, n>N$ 时, 有

$$
\begin{aligned}
\left|x_{n}-x_{m}\right| & =\left|\left(x_{n}-a\right)-\left(x_{m}-a\right)\right| \\
& \leqslant\left|x_{n}-a\right|+\left|x_{n}-a\right|<\frac{\varepsilon}{2}+\frac{\varepsilon}{2}=\varepsilon,
\end{aligned}
$$

所以条件是必要的.

充分性这里不予证明.

这准则的几何意义表示, 数列 $\left\{x_{n}\right\}$ 收敛的充分必要条件是: 对于任意给定 的正数 $\varepsilon$, 在数轴上一切具有足够大号码的点 $x_{n}$ 中, 任意两点间的距离小于 $\varepsilon$.

柯西极限存在准则有时也叫做柯西审敛原理.

## 习 题 1-6

1. 计算下列极限:
(1) $\lim _{x \rightarrow 0} \frac{\sin \omega x}{x}$;
(2) $\lim _{x \rightarrow 11} \frac{\tan 3 x}{x}$;
(3) $\lim _{x \rightarrow 0} \frac{\sin 2 x}{\sin 5 x}$;
(4) $\lim _{x \rightarrow 10} x \cot x$;
(5) $\lim _{x \rightarrow 0} \frac{1-\cos 2 x}{x \sin x}$;
(6) $\lim _{n \rightarrow \infty} 2^{n} \sin \frac{x}{2^{n}}$ ( $x$ 为不等于零的常数).
2. 计算下列极限:
(1) $\lim _{x \rightarrow 0}(1-x)^{\frac{1}{x}}$;
(2) $\lim _{x \rightarrow 0}(1+2 x)^{\frac{1}{x}}$;
(3) $\lim _{x \rightarrow \infty}\left(\frac{1+x}{x}\right)^{2 x}$;
(4) $\lim _{x \rightarrow \infty}\left(1-\frac{1}{x}\right)^{k \cdot r}(k$ 为正整数).

3. 根据函数极限的定义, 证明极限存在的准则 I

4. 利用极限存在准则证明:
(1) $\lim _{n \rightarrow \infty} \sqrt{1+\frac{1}{n}}=1$;
(2) $\lim _{n \rightarrow \infty} n\left(\frac{1}{n^{2}+\pi}+\frac{1}{n^{2}+2 \pi}+\cdots+\frac{1}{n^{2}+n \pi}\right)=1$; (3) 数列 $\sqrt{2}, \sqrt{2+\sqrt{2}}, \sqrt{2+\sqrt{2+\sqrt{2}}}, \cdots$ 的极限存在;

(4) $\lim _{x \rightarrow 0} \sqrt[n]{1+x}=1$;

(5) $\lim _{x \rightarrow 1^{+}} x\left[\frac{1}{x}\right]=1$.

## 第七节 无穷小的比较

在第五节中我们已经知道,两个无穷小的和、差及乘积仍旧是无穷小. 但是, 关于两个无穷小的商, 却会出现不同的情况, 例如, 当 $x \rightarrow 0$ 时, $3 x 、 x^{2} 、 \sin x$ 都 是无穷小, 而

$$
\lim _{x \rightarrow 0} \frac{x^{2}}{3 x}=0, \quad \lim _{x \rightarrow 0} \frac{3 x}{x^{2}}=\infty, \quad \lim _{x \rightarrow 0} \frac{\sin x}{3 x}=\frac{1}{3} .
$$

两个无穷小之比的极限的各种不同情况, 反映了不同的无穷小趋于零的 “快慢” 程度. 就上面几个例子来说, 在 $x \rightarrow 0$ 的过程中, $x^{2} \rightarrow 0$ 比 $3 x \rightarrow 0$ “快些”, 反过来 $3 x \rightarrow 0$ 比 $x^{2} \rightarrow 0$ “慢些”, 而 $\sin x \rightarrow 0$ 与 $x \rightarrow 0$ “快慢相仿”.

下面, 我们就无穷小之比的极限存在或为无穷大时, 来说明两个无穷小之间 的比较. 应当注意,下面的 $\alpha$ 及 $\beta$ 都是在同一个自变量的变化过程中的无穷小, 且 $\alpha \neq 0, \lim \frac{\beta}{\alpha}$ 也是在这个变化过程中的极限.

定义:

如果 $\lim \frac{\beta}{\alpha}=0$, 就说 $\beta$ 是比 $\alpha$ 高阶的无穷小,记作 $\beta=o(\alpha)$;

如果 $\lim \frac{\beta}{\alpha}=\infty$, 就说 $\beta$ 是比 $\alpha$ 低阶的无穷小.

如果 $\lim \frac{\beta}{\alpha}=c \neq 0$, 就说 $\beta$ 与 $\alpha$ 是同阶无穷小;

如果 $\lim \frac{\beta}{\alpha^{k}}=c \neq 0, k>0$, 就说 $\beta$ 是关于 $\alpha$ 的 $k$ 阶无穷小.

如果 $\lim \frac{\beta}{\alpha}=1$, 就说 $\beta$ 与 $\alpha$ 是等价无穷小, 记作 $\alpha-\beta$.

显然,等价无穷小是同阶无穷小的特殊情形, 即 $c=1$ 的情形.

下面举一些例子:

因为 $\lim _{x \rightarrow 0} \frac{3 x^{2}}{x}=0$, 所以当 $x \rightarrow 0$ 时, $3 x^{2}$ 是比 $x$ 高阶的无穷小, 即 $3 x^{2}=$ $o(x)(x \rightarrow 0)$. 因为 $\lim _{n \rightarrow \infty} \frac{\frac{1}{n}}{\frac{1}{n^{2}}}=\infty$, 所以当 $n \rightarrow \infty$ 时, $\frac{1}{n}$ 是比 $\frac{1}{n^{2}}$ 低阶的无穷小.

因为 $\lim _{x \rightarrow 3} \frac{x^{2}-9}{x-3}=6$, 所以当 $x \rightarrow 3$ 时, $x^{2}-9$ 与 $x-3$ 是同阶无穷小.

因为 $\lim _{x \rightarrow 0} \frac{1-\cos x}{x^{2}}=\frac{1}{2}$, 所以当 $x \rightarrow 0$ 时, $1-\cos x$ 是关于 $x$ 的二阶无穷小.

因为 $\lim _{x \rightarrow 0} \frac{\sin x}{x}=1$, 所以当 $x \rightarrow 0$ 时, $\sin x$ 与 $x$ 是等价无穷小, 即 $\sin x \sim$ $x(x \rightarrow 0)$.

下面再举二个常用的等价无穷小的例子.

例 1 证明: 当 $x \rightarrow 0$ 时, $\sqrt[n]{1+x}-1 \sim \frac{1}{n} x$.

证因为

$$
\begin{aligned}
\lim _{x \rightarrow 0} \frac{\sqrt[n]{1+x}-1}{\frac{1}{n} x} & =\lim _{x \rightarrow 0} \frac{(\sqrt[n]{1+x})^{n}-1}{\frac{1}{n} x\left[\sqrt[n]{(1+x)^{n-1}}+\sqrt[n]{(1+x)^{n-2}}+\cdots+1\right]} \\
& =\lim _{x \rightarrow 0} \frac{n}{\sqrt[n]{(1+x)^{n-1}}+\sqrt[n]{(1+x)^{n-2}}+\cdots+1}=1
\end{aligned}
$$

所以 $\sqrt[n]{1+x}-1-\frac{1}{n} x(x \rightarrow 0)$.

关于等价无穷小,有下面两个定理.

定理 $1 \beta$ 与 $\alpha$ 是等价无穷小的充分必要条件为

$$
\beta=\alpha+o(\alpha) \text {. }
$$

证 必要性 设 $\alpha \sim \beta$, 则

$$
\lim \frac{\beta-\alpha}{\alpha}=\lim \left(\frac{\beta}{\alpha}-1\right)=\lim \frac{\beta}{\alpha}-1=0,
$$

因此 $\beta-\alpha=o(\alpha)$, 即 $\beta=\alpha+o(\alpha)$.

充分性 设 $\beta=\alpha+o(\alpha)$, 则

$$
\lim \frac{\beta}{\alpha}=\lim \frac{\alpha+o(\alpha)}{\alpha}=\lim \left(1+\frac{o(\alpha)}{\alpha}\right)=1,
$$

因此 $\alpha \sim \beta$.

例 2 因为当 $x \rightarrow 0$ 时, $\sin x \sim x, \tan x \sim x, \arcsin x \sim x, 1=\cos x \sim \frac{1}{2} x^{2}$,

(1) 极限 $\lim _{r \rightarrow 0} \sqrt[n]{(1+x)^{m}}=1(m=n-1, n-2, \cdots, 1)$ 川1到了可题 $1-6$ 中题 4(4) 的炶果及第五节中 定理 3 的推论 2 . 所以当 $x \rightarrow 0$ 时有

$$
\sin x=x+o(x), \quad \tan x=x+o(x),
$$

$\arcsin x=x+o(x), \quad 1-\cos x=\frac{1}{2} x^{2}+o\left(x^{2}\right)$.

定理 2 设 $\alpha \sim \alpha^{\prime}, \beta \sim \beta^{\prime}$, 且 $\lim \frac{\beta^{\prime}}{\alpha}$ 存在, 则

$$
\lim \frac{\beta}{\alpha}=\lim \frac{\beta^{\prime}}{\alpha} .
$$

证 $\lim \frac{\beta}{\alpha}=\lim \left(\frac{\beta}{\beta^{\prime}} \cdot \frac{\beta^{\prime}}{\alpha} \cdot \frac{\alpha^{\prime}}{\alpha}\right)$

$$
=\lim \frac{\beta}{\beta^{\prime}} \cdot \lim \frac{\beta^{\prime}}{\alpha} \cdot \lim \frac{\alpha^{\prime}}{\alpha}=\lim \frac{\beta^{\prime}}{\alpha} .
$$

定理 2 表明,求两个无穷小之比的极限时, 分子及分母都可用等价无穷小来 代替. 因此,如果用来代替的无穷小选得适当的话, 可以使计算简化.

例 3 求 $\lim _{x \rightarrow 0} \frac{\tan 2 x}{\sin 5 x}$.

解 当 $x \rightarrow 0$ 时, $\tan 2 x \sim 2 x, \sin 5 x \sim 5 x$, 所以

$$
\lim _{x \rightarrow 0} \frac{\tan 2 x}{\sin 5 x}=\lim _{x \rightarrow 0} \frac{2 x}{5 x}=\frac{2}{5} \text {. }
$$

例 4 求 $\lim _{x \rightarrow 0} \frac{\sin x}{x^{3}+3 x}$.

解 当 $x \rightarrow 0$ 时, $\sin x \sim x$, 无穷小 $x^{3}+3 x$ 与它本身显然是等价的, 所以

$$
\lim _{x \rightarrow 0} \frac{\sin x}{x^{3}+3 x}=\lim _{x \rightarrow 4} \frac{x}{x\left(x^{2}+3\right)}=\lim _{x \rightarrow 0} \frac{1}{x^{2}+3}=\frac{1}{3} \text {. }
$$

例 5 求 $\lim _{x \rightarrow 0} \frac{\left(1+x^{2}\right)^{\frac{1}{3}}-1}{\cos x-1}$.

解 当 $x \rightarrow 0$ 时, $\left(1+x^{2}\right)^{\frac{1}{3}}-1-\frac{1}{3} x^{2}, \cos x-1--\frac{1}{2} x^{2}$, 所以

$$
\lim _{x \rightarrow 0} \frac{\left(1+x^{2}\right)^{\frac{1}{3}}-1}{\cos x-1}=\lim _{x \rightarrow 0} \frac{\frac{1}{3} x^{2}}{-\frac{1}{2} x^{2}}=-\frac{2}{3} .
$$

## 习 题 $1-7$

1. 当 $x \rightarrow 0$ 时, $2 x-x^{2}$ 与 $x^{2}-x^{3}$ 相比，哪一个是高阶无穷小?
2. 当 $x \rightarrow 1$ 时,无穷小 $1-x$ 和(1) $1-x^{3}$, (2) $\frac{1}{2}\left(1-x^{2}\right)$ 是否同阶? 是否等价?
3. 证明: 当 $x \rightarrow 0$ 时,有
(1) $\arctan x \sim x$;
(2) $\sec x-1-\frac{x^{2}}{2}$.
4. 利用等价无穷小的性质,求下列极限：
(1) $\lim _{x \rightarrow 11} \frac{\tan 3 x}{2 x}$;
(2) $\lim _{x \rightarrow 0} \frac{\sin \left(x^{n}\right)}{(\sin x)^{m}}(n, m$ 为正整数);
(3) $\lim _{x \rightarrow 10} \frac{\tan x-\sin x}{\sin ^{3} x}$;
(4) $\lim _{x \rightarrow 11} \frac{\sin x-\tan x}{\left(\sqrt[3]{1+x^{2}}-1\right)(\sqrt{1+\sin x}-1)}$.
5. 证明无穷小的等价关系具有下列性质:
(1) $a \sim \alpha$ (自反性);
（2）若 $\alpha \sim \beta$, 则 $\beta-\alpha$ (对称性);

(3) 若 $a-\beta, \beta \sim \gamma$, 则 $\alpha \sim \gamma$ (传递性).

## 第八节 函数的连续性与间断点

## 一、函数的连续性

自然界中有许多现象，如气温的变化、河水的流动、植物的生长等等,都是连 续地变化着的. 这种现象在函数关系上的反咉, 就是函数的连续性. 例如就气温 的变化来看, 当时间变动很微小时, 气温的变化也很微小, 这种特点就是所谓连 续性.下面我们先引人增量的概念, 然后来描述连续性, 并引出函数的连续性的 定义.

设变量 $u$ 从它的一个初值 $u_{1}$ 变到终值 $u_{2}$, 终值与初值的差 $u_{2}-u_{1}$ 就叫 做变量 $u$ 的增量, 记作 $\Delta u$, 即

$$
\Delta u=u_{2}-u_{1} .
$$

$u_{2}=u_{1}+\Delta u$ 时是增大的; 当 $\Delta u$ 为负时, 变量 $u$ 是减小的.

应该注意到: 记号 $\Delta u$ 并不表示某个量 $\Delta$ 与变量 $u$ 的乘积,而是一个整体 不可分割的记号.

现在假定函数 $y=f(x)$ 在点 $x_{0}$ 的某一个邻域内是有定义的. 当自变量 $x$ 在这邻域内从 $x_{0}$ 变到 $x_{0}+\Delta x$ 时, 函数 $y$ 相应的从 $f\left(x_{0}\right)$ 变到 $f\left(x_{0}+\Delta x\right)$, 因此函数 $y$ 的对应增量为

$$
\Delta y=f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right) \text {. }
$$

这个关系式的几何解释如图 1-35 所示.

假如保持 $x_{10}$ 不变而让自变黑的增量 $\Delta x$ 变动, 一般说来, 函数 $y$ 的增量 $\Delta y$ 也要随着变动. 现在我 们对连续性的概念可以这样描述: 如果当 $\Delta x$ 趋于

图 $1-35$ 零时, 函数 $y$ 的对应增量 $\Delta y$ 也趋于零, 即

$$
\lim _{\Delta x \rightarrow 0} \Delta y=0
$$

或

$$
\lim _{\Delta r \rightarrow 01}\left[f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)\right]=0,
$$

那么就称函数 $y=f(x)$ 在点 $x_{0}$ 处是连续的, 即有下述定义:

定义 设函数 $y=f(x)$ 在点 $x_{0}$ 的某一邻域内有定义, 如果

$$
\lim _{\Delta x \rightarrow 0} \Delta y=\lim _{\Delta x \rightarrow 0}\left[f\left(x_{11}+\Delta x\right)-f\left(x_{0}\right)\right]=0,
$$

那么就称函数 $y=f(x)$ 在点 $x_{0}$ 连续.

为了应用方便起见,下面把函数 $y=f(x)$ 在点 $x_{0}$ 连续的定义用不同的方 式来叙述.

设 $x=x_{0}+\Delta x$, 则 $\Delta x \rightarrow 0$ 就是 $x \rightarrow x_{0}$. 又由于

$$
\Delta y=f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)=f(x)-f\left(x_{0}\right)
$$

即

$$
f(x)=f\left(x_{0}\right)+\Delta y,
$$

可见 $\Delta y \rightarrow 0$ 就是 $f(x) \rightarrow f\left(x_{0}\right)$, 因此 (1) 式与

$$
\lim _{x \rightarrow x_{0}} f(x)=f\left(x_{0}\right)
$$

相当. 所以, 函数 $y=f(x)$ 在点 $x_{0}$ 连续的定义又可叙述如下:

设函数 $y=f(x)$ 在点 $x_{0}$ 的某一邻域内有定义, 如果

$$
\lim _{x \rightarrow x_{11}} f(x)=f\left(x_{11}\right),
$$

那么就称函数 $f(x)$ 在点 $x_{0}$ 连续.

由函数 $f(x)$ 当 $x \rightarrow x_{0}$ 时的极限的定义可知, 上述定义也可用 “ $\varepsilon-\delta$ ”语言 表达如下:

$f(x)$ 在点 $x_{0}$ 连续 $\Leftrightarrow \forall \varepsilon>0, \exists \delta>0$, 当 $\left|x-x_{0}\right|<\delta$ 时, 有 $\left|f(x)-f\left(x_{0}\right)\right|<\varepsilon$.

下面说明左连续及右连续的概念.

如果 $\lim _{x \rightarrow x_{0}^{-}} f(x)=f\left(x_{0}^{-}\right)$存在且等于 $f\left(x_{0}\right)$, 即

$$
f\left(x_{0}^{-}\right)=f\left(x_{0}\right),
$$

就说函数 $f(x)$ 在点 $x_{0}$ 左连续. 如果 $\lim _{x \rightarrow x_{0}^{+}} f(x)=f\left(x_{0}^{\prime}\right)$ 存在且等于 $f\left(x_{0}\right)$, 即

$$
f\left(x_{0}^{+}\right)=f\left(x_{0}\right),
$$

就说函数 $f(x)$ 在点 $x_{0}$ 右连续.

在区间上每一点都连续的函数, 叫做在该区间上的连续函数, 或者说函数在 该区间上连续. 如果区间包括端点, 那么函数在右端点连续是指左连续, 在左端 点连续是指右连续. 连续函数的图形是一条连续而不间断的曲线.

在第五节中,我们曾经证明: 如果 $f(x)$ 是有理整函数 (多项式), 则对于任 意的实数 $x_{11}$, 都有 $\lim _{x \rightarrow x_{\mathrm{n}}} f(x)=f\left(x_{0}\right)$, 因此有理整函数在区间 $(-\infty,+\infty)$ 内是 连续的. 对于有理分式函数 $F(x)=\frac{P(x)}{Q(x)}$, 只要 $Q\left(x_{0}\right) \neq 0$, 就有 $\lim _{x \rightarrow r_{0}} F(x)=$ $F\left(x_{0}\right)$ ，因此有理分式函数在其定义域内的每一点都是连续的.

由第三节例 5 可知, 函数 $f(x)=\sqrt{x}$ 在 $(0,+\infty)$ 内是连续的.

作为例子,我们来证明, 函数 $y=\sin x$ 在区间 $(-\infty,+\infty)$ 内是连续的.

设 $x$ 是区间 $(-\infty,+\infty)$ 内任意取定的一点. 当 $x$ 有增量 $\Delta x$ 时,对应的函 数的增量为

$$
\Delta y=\sin (x+\Delta x)-\sin x,
$$

由三角公式有

$$
\sin (x+\Delta x)-\sin x=2 \sin \frac{\Delta x}{2} \cos \left(x+\frac{\Delta x}{2}\right),
$$

注意到

$$
\left|\cos \left(x+\frac{\Delta x}{2}\right)\right| \leqslant 1
$$

就推得

$$
|\Delta y|=|\sin (x+\Delta x)-\sin x| \leqslant 2\left|\sin \frac{\Delta x}{2}\right| .
$$

因为对于任意的角度 $\alpha$, 当 $\alpha \neq 0$ 时有 $|\sin \alpha|<|\alpha|$, 所以

$$
0 \leqslant|\Delta y|=|\sin (x+\Delta x)-\sin x|<|\Delta x| \text {. }
$$

因此, 当 $\Delta x \rightarrow 0$ 时, 由夹逼准则得 $|\Delta y| \rightarrow 0$, 这就证明了 $y=\sin x$ 对于任一 $x \in(-\infty,+\infty)$ 是连续的.

类似地可以证明,函数 $y=\cos x$ 在区间 $(-\infty,+\infty)$ 内是连续的.

## 二、函数的间断点

设函数 $f(x)$ 在点 $x_{0}$ 的某去心邻域内有定义. 在此前提下, 如果函数 $f(x)$ 有下列三种情形之一:

(1) 在 $x=x_{0}$ 没有定义;

（2）虽在 $x=x_{0}$ 有定义, 但 $\lim _{x \rightarrow x_{10}} f(x)$ 不存在;

（3）虽在 $x=x_{0}$ 有定义，且 $\lim _{x \rightarrow x_{0}} f(x)$ 存在，但 $\lim _{x \rightarrow x_{0}} f(x) \neq f\left(x_{0}\right)$,

则函数 $f^{\prime}(x)$ 在点 $x_{11}$ 为不连续, 而点 $x_{0}$ 称为函数 $f(x)$ 的不连续点或间断点.

下面举例来说明函数间断点的几种常见类型. 例 1 正切函数 $y=\tan x$ 在 $x=\frac{\pi}{2}$ 处没有定义, 所以点 $x=\frac{\pi}{2}$ 是函数 $\tan x$ 的间断点. 因

$$
\lim _{x \rightarrow \frac{\pi}{2}} \tan x=\infty,
$$

我们称 $x=\frac{\pi}{2}$ 为函数 $\tan x$ 的无穷间断点 (图 1-36).

例 2 函数 $y=\sin \frac{1}{x}$ 在点 $x=0$ 没有定义; 当 $x \rightarrow 0$ 时, 函数值在 -1 与 +1 之间变动无限多次 (图 1-37), 所以点 $x=0$ 称为函数 $\sin \frac{1}{x}$ 的振荡间断点.

图 $\quad 1-37$

例 3 函数 $y=\frac{x^{2}-1}{x-1}$ 在点 $x=1$ 没有定义, 所以函数在点 $x=1$ 为不连续 (图 1-38). 但这里

$$
\lim _{x \rightarrow 1} \frac{x^{2}-1}{x-1}=\lim _{x \rightarrow 1}(x+1)=2 .
$$

如果补充定义: 令 $x=1$ 时 $y=2$, 则所给函数在 $x=1$ 成 为连续. 所以 $x=1$ 称为该函数的可去间断点.

例 4 函数

$$
y=f(x)= \begin{cases}x, & x \neq 1, \\ \frac{1}{2}, & x=1 .\end{cases}
$$

国 $1-38$

这里 $\lim _{x \rightarrow 1} f(x)=\lim _{x \rightarrow 1} x=1$, 但 $f(1)=\frac{1}{2}$, 所以

$$
\lim _{x \rightarrow 1} f(x) \neq f(1) .
$$

因此, 点 $x=1$ 是函数 $f(x)$ 的间断点 (图 1-39). 但如果改变函数 $f(x)$ 在 $x=1$ 处的定义: 令 $f(1)=1$, 则 $f(x)$ 在 $x=1$ 成为连续. 所以 $x=1$ 也称为该函数的 可去间断点.

例 5 函数

这里, 当 $x \rightarrow 0$ 时,

$$
f(x)= \begin{cases}x-1, & x<0, \\ 0, & x=0, \\ x+1 & x>0 .\end{cases}
$$

$$
\begin{gathered}
\lim _{x \rightarrow 0^{-}} f(x)=\lim _{x \rightarrow 0^{-}}(x-1)=-1, \\
\lim _{x \rightarrow 1^{!}} f(x)=\lim _{x \rightarrow 1^{+}}(x+1)=1 .
\end{gathered}
$$

左极限与右极限虽都存在, 但不相等, 故极限 $\lim _{x \rightarrow 0} f(x)$ 不存在, 所以点 $x=0$ 是函 数 $f(x)$ 的间断点 (图 1-40). 因 $y=f(x)$ 的图形在 $x=0$ 处产生跳跃现象, 我 们称 $x=0$ 为函数 $f(x)$ 的跳跃间断点.

因 1 - 39

因 $1-40$

上面举了一些间断点的例子，通常把间断点分成两类: 如果 $x_{0}$ 是函数 $f(x)$ 的间断点, 但左极限 $f\left(x_{10}{ }^{-}\right)$及右极限 $f\left(x_{11}^{\prime}\right)$ 都存在, 那么 $x_{01}$ 称为函数 $f(x)$ 的第一类间断点. 不是第一类间断点的任何间断点, 称为第二类间断点. 在 第一类间断点中,左、右极限相等者称为可去间断点, 不相等者称为跳跃间断点. 无穷间断点和振荡间断点显然是第二类间断点.

## 习 题 $1-8$

1. 设 $y=f(x)$ 的图形如图 1-41 所示, 试指出 $f(x)$ 的全部间断点,并对可去间断点补 充或修改函数值的定义，使它成为连续点. 2. 研究下列函数的连续性,并画出函数的图形:

(1) $f(x)= \begin{cases}x^{2}, & 0 \leqslant x \leqslant 1, \\ 2-x, & 1<x \leqslant 2 ;\end{cases}$

(2) $f(x)= \begin{cases}x, & -1 \leqslant x \leqslant 1, \\ 1, & x<-1 \text { 或 } x>1 .\end{cases}$

3. 下列函数在指出的点处间断, 说明这些间断点瓜于 哪一类, 如果是可去间断点, 则补充或改变函数的定义使它 连续:

(1) $y=\frac{x^{2}-1}{x^{2}-3 x+2}, x=1, x=2$;

(2) $y=\frac{x}{\tan x}, x=k \pi, x=k \pi+\frac{\pi}{2} \quad(k=0, \pm 1, \pm 2 \cdots)$;

(3) $y=\cos ^{2} \frac{1}{x}, x=0$;

(4) $y=\left\{\begin{array}{ll}x-1, & x \leqslant 1, \\ 3-x, & x>1,\end{array} \quad x=1\right.$.

4. 讨论函数 $f(x)=\lim _{4 \rightarrow \infty} \frac{1-x^{2 n}}{1+x^{2 n}} x$ 的连续性, 若有间断点, 判别其类型.
5. 下列陈述中, 哪些是对的, 哪些是错的? 如果是对的, 说明理由; 如果是错的, 试给出 一个反例.

(1) 如果函数 $f(x)$ 在 $a$ 连续,那么 $|f(x)|$ 也在 $a$ 连续;

(2) 如果函数 $|f(x)|$ 在 $a$ 连续, 那么 $f(x)$ 地在 $a$ 连续.

6. 证明: 若函数 $f(x)$ 在点 $x_{0}$ 连续且 $f\left(x_{0}\right) \neq 0$, 则存在 $x_{01}$ 的某一邻域 $U\left(x_{0}\right)$, 当 $x \in U\left(x_{0}\right)$ 时, $f(x) \neq 0$.

7. 设

$$
f(x)= \begin{cases}x, & x \in \mathbf{Q}, \\ 0, & x \in \mathbf{R} \backslash \mathbf{Q} .\end{cases}
$$

证明:

(1) $f(x)$ 在 $x=0$ 连续;

(2) $f(x)$ 在非零的 $x$ 处都不连续.

8. 试举出具有以下性质的函数 $f(x)$ 的例子:

$x=0, \pm 1, \pm 2, \pm \frac{1}{2}, \cdots, \pm n, \pm \frac{1}{n}, \cdots$ 是 $f(x)$ 的所有间断点, 且它们都是无穷间断点.

## 第九节 连续函数的运算与初等函数的连续性

## 一、连续函数的和、差、积、商的连续性

由函数在某点连续的定义和极限的四则运算法则,立即可得出下面的定理. 定理 1 设函数 $f(x)$ 和 $g(x)$ 在点 $x_{0}$ 连续, 则它们的和 (差) $f \pm g$ 、积 $f \cdot g$ 及商 $\frac{f}{g}$ (当 $g\left(x_{0}\right) \neq 0$ 时) 都在点 $x_{0}$ 连续.

例 1 因 $\tan x=\frac{\sin x}{\cos x}, \cot x=\frac{\cos x}{\sin x}$, 而 $\sin x$ 和 $\cos x$ 都在区间 $(-\infty$, $+\infty)$ 内连续(第八节), 故由定理 1 知 $\tan x$ 和 $\cot x$ 在它们的定义域内是连续 的.

## 二、反函数与复合函数的连续性

反函数和复合函数的概念已经在第一节中讲过,这里来讨论它们的连续性.

定理 2 如果函数 $y=f(x)$ 在区间 $I_{x}$ 上单调增加(或单调减少)且连续, 那 么它的反函数 $x=f^{-1}(y)$ 也在对应的区间 $I_{y}=\left\{y \mid y=f(x), x \in I_{x}\right\}$ 上单调增 加(或单调减少)且连续.

证明从略.

例 2 由于 $y=\sin x$ 在闭区间 $\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ 上单调增加且连续,所以它的反 函数 $y=\arcsin x$ 在闭区间 $[-1,1]$ 上也是单调增加且连续的.

同样,应用定理 2 可证: $y=\arccos x$ 在闭区间 $[-1,1]$ 上单调减少且连续; $y=\arctan x$ 在区间 $(-\infty,+\infty)$ 内单调增加且连续; $y=\operatorname{arccot} x$ 在区间 $(-\infty$, $+\infty)$ 内单调减少且连续.

总之,反三角函数 $\arcsin x, \arccos x, \arctan x, \operatorname{arccot} x$ 在它们的定义域内 都是连续的.

定理 3 设函数 $y=f[g(x)]$ 由函数 $u=g(x)$ 与函数 $y=f(u)$ 复合而成， $\stackrel{O}{U}\left(x_{0}\right) \subset D_{f_{0} k}$. 若 $\lim _{x \rightarrow x_{0}} g(x)=u_{0}$, 而函数 $y=f(u)$ 在 $u=u_{0}$ 连续, 则

$$
\lim _{x \rightarrow x_{u}} f[g(x)]=\lim _{n \rightarrow u_{0}} f(u)=f\left(u_{0}\right) .
$$

证 在第五节定理 6 中, 令 $A=f\left(u_{0}\right)$ (这时 $f(u)$ 在点 $u_{0}$ 连续), 并取消 “存在 $\delta_{0}>0$, 当 $x \in \stackrel{U}{U}\left(x_{0}, \delta_{0}\right)$ 时, 有 $g(x) \neq u_{0}$ ”这条件, 便得上面的定理. 这 里 $g(x) \neq u_{0}$ 这条件可以取消的理由是: $\forall \varepsilon>0$, 使 $g(x)=u_{0}$ 成立的那些点 $x$, 显然也使 $\left|f[g(x)]-f\left(u_{0}\right)\right|<\varepsilon$ 成立. 因此附加 $g(x) \neq u_{0}$ 这条件就没有 必要了.

因为在定理 3 中有

故(1)式又可写成

$$
\lim _{x \rightarrow x_{0}} g(x)=u_{0}, \quad \lim _{u \rightarrow u_{0}} f(u)=f\left(u_{0}\right),
$$

$$
\lim _{x \rightarrow x_{0}} f[g(x)]=f\left[\lim _{x \rightarrow x_{0}} g(x)\right] .
$$

(1)式表示, 在定理 3 的条件下, 如果作代换 $u=g(x)$, 那么求 $\lim _{x \rightarrow x_{0}} f[g(x)]$ 就化为求 $\lim _{u \rightarrow u_{0}} f(u)$, 这里 $u_{u}=\lim _{x \rightarrow r_{0}} g(x)$.

(2) 式表示, 在定理 3 的条件下, 求复合函数 $f[g(x)]$ 的极限时, 函数符号 $f$ 与极限号 $\lim _{x \rightarrow x_{0}}$ 可以交换次序.

把定理 3 中的 $x \rightarrow x_{0}$ 换成 $x \rightarrow \infty$, 可得类似的定理.

例 3 求 $\lim _{x \rightarrow 3} \sqrt{\frac{x-3}{x^{2}-9}}$.

解 $y=\sqrt{\frac{x-3}{x^{2}-9}}$ 可看作由 $y=\sqrt{u}$ 与 $u=\frac{x-3}{x^{2}-9}$ 复合而成. 因为 $\lim _{x \rightarrow 3} \frac{x-3}{x^{2}-9}=\frac{1}{6}$, 而函数 $y=\sqrt{u}$ 在点 $u=\frac{1}{6}$ 连续, 所以

$$
\lim _{x \rightarrow 3} \sqrt{\frac{x-3}{x^{2}-9}}=\sqrt{\lim _{x \rightarrow 3} \frac{x-3}{x^{2}-9}}=\sqrt{\frac{1}{6}}=\frac{\sqrt{6}}{6} .
$$

定理 4 设函数 $y=f[g(x)]$ 是由函数 $u=g(x)$ 与函数 $y=f(u)$ 复合而 成, $U\left(x_{0}\right) \subset D_{f \circ \mathrm{g}}$. 若函数 $u=g(x)$ 在 $x=x_{0}$ 连续, 且 $g\left(x_{0}\right)=u_{0}$, 而函数 $y=$ $f(u)$ 在 $u=u_{0}$ 连续,则复合函数 $y=f[g(x)]$ 在 $x=x_{0}$ 也连续.

证 只要在定理 3 中令 $u_{0}=g\left(x_{0}\right)$, 这就表示 $g(x)$ 在点 $x_{0}$ 连续, 于是由 (1) 式得

$$
\lim _{x \rightarrow x_{11}} f[g(x)]=f\left(u_{11}\right)=f\left[g\left(x_{11}\right)\right]^{2},
$$

这就证明了复合函数 $f[g(x)]$ 在点 $x_{0}$ 连续.

例 4 讨论函数 $y=\sin \frac{1}{x}$ 的连续性.

解 函数 $y=\sin \frac{1}{x}$ 可看作是由 $u=\frac{1}{x}$ 及 $y=\sin u$ 复合而成的. $\frac{1}{x}$ 当 $-\infty<$ $x<0$ 和 $0<x<+\infty$ 时是连续的, $\sin u$ 当 $-\infty<u<+\infty$ 时是连续的. 根据定理 4 , 函数 $\sin \frac{1}{x}$ 在无限区间 $(-\infty, 0)$ 和 $(0,+\infty)$ 内是连续的.

## 三、初等函数的连续性

前面证明了三角函数及反三角函数在它们的定义域内是连续的.

我们指出(但不详细讨论), 指数函数 $a^{x}(a>0, a \neq 1)$ 对于一切实数 $x$ 都 有定义, 且在区间 $(-\infty,+\infty)$ 内是单调的和连续的, 它的值域为 $(0,+\infty)$.

由指数函数的单调性和连续性, 引用定理 2 可得: 对数函数 $\log _{a} x(a>0$, $a \neq 1)$ 在区间 $(0,+\infty)$ 内单调且连续.

幂函数 $y=x^{\prime \prime}$ 的定义域随 $\mu$ 的值而异,但无论 $\mu$ 为何值,在区间 $(0,+\infty)$ 内幂函数总是有定义的.下面我们来证明,在 $(0,+\infty)$ 内幂函数是连续的. 事实 上,设 $x>0$, 则

$$
y=x^{\prime \prime}=a^{\text {rlthex }} .^{\prime \prime},
$$

因此,幂函数 $x^{\prime \prime}$ 可看作是由 $y=a^{\prime \prime}, u=\mu \log _{a} x$ 复合而成的, 由此,根据定理 4, 它在 $(0,+\infty)$ 内连续. 如果对于 $\mu$ 取各种不同值加以分别讨论, 可以证明 (证明 从略)幂函数在它的定义域内是连续的.

综合起来得到: 基本初等函数在它们的定义域内都是连续的.

最后, 根据第一节中关于初等函数的定义, 由基本初等函数的连续性以及本 节定理 1 、定理 4 可得下列重要结论: 一切初等函数在其定义区间内都是连续 的.所谓定义区间, 就是包含在定义域内的区间.

根据函数 $f(x)$ 在点 $x_{10}$ 连续的定义, 如果已知 $f(x)$ 在点 $x_{0}$ 连续, 那么求 $f(x)$ 当 $x \rightarrow x_{0}$ 的极限时, 只要求 $f(x)$ 在点 $x_{0}$ 的函数值就行了. 因此, 上述关 于初等函数连续性的结论提供了求极限的一个方法, 这就是: 如果 $f(x)$ 是初等 函数, 且 $x_{0}$ 是 $f(x)$ 的定义区间内的点, 则

$$
\lim _{x \rightarrow x_{0}} f(x)=f\left(x_{0}\right) \text {. }
$$

例如, 点 $x_{0}=0$ 是初等函数 $f(x)=\sqrt{1-x^{2}}$ 的定义区间 $[-1,1]$ 内的点,所以 $\lim _{x \rightarrow 0} \sqrt{1-x^{2}}=\sqrt{1}=1$; 又如点 $x_{0}=\frac{\pi}{2}$ 是初等函数 $f(x)=\ln \sin x$ 的一个定义区 间 $(0, \pi)$ 内的点, 所以

$$
\lim _{x \rightarrow \frac{\pi}{2}} \ln \sin x=\ln \sin \frac{\pi}{2}=0 .
$$

例 5 求 $\lim _{x \rightarrow 0} \frac{\sqrt{1+x^{2}}-1}{x}$.

$$
\text { 解 } \begin{aligned}
\lim _{x \rightarrow 0} \frac{\sqrt{1+x^{2}}-1}{x} & =\lim _{x \rightarrow 0} \frac{\left(\sqrt{1+x^{2}}-1\right)\left(\sqrt{1+x^{2}}+1\right)}{x\left(\sqrt{1+x^{2}}+1\right)} \\
& =\lim _{x \rightarrow 0} \frac{x}{\sqrt{1+x^{2}}+1}=\frac{0}{2}=0 .
\end{aligned}
$$

例 6 求 $\lim _{x \rightarrow 0} \frac{\log _{u}(1+x)}{x}$.

解 $\lim _{x \rightarrow 0} \frac{\log _{a}(1+x)}{x}=\lim _{x \rightarrow 1} \log _{a}(1+x)^{\frac{1}{x}}=\log _{a} \mathrm{e}=\frac{1}{\ln a}$.

例 7 求 $\lim _{x \rightarrow 10} \frac{a^{x}-1}{x}$.

解 令 $a^{x}-1=t$, 则 $x=\log _{a}(1+t)$, 当 $x \rightarrow 0$ 时 $t \rightarrow 0$, 于是

$$
\lim _{x \rightarrow 0} \frac{a^{x}-1}{x}=\lim _{t \rightarrow 0} \frac{t}{\log _{a}(1+t)}=\ln a .
$$

例 8 求 $\lim _{x \rightarrow 0}(1+2 x)^{\frac{3}{\sin x}}$.

解 因为

利用定理 3 及极限的运算法则, 便有

$$
(1+2 x)^{\frac{3}{x+11 x}}=(1+2 x)^{\frac{1}{2 x} \cdot \frac{x}{\sin x} \cdot 6}=\mathrm{e}^{6 \cdot \frac{x}{\sin \cdot x^{2}} \ln (1+2 .)^{\frac{1}{2 \cdot x}}},
$$

$$
\lim _{x \rightarrow 0}(1+2 x)^{\frac{3}{\sin . x}}=\mathrm{e}^{\lim ^{\lim _{x \rightarrow 1}}\left[0 \cdot \frac{1}{\sin . x} \cdot \ln (1+2 x)^{\frac{1}{2 x}}\right]}=\mathrm{e}^{0} .
$$

一般的, 对于形如 $u(x)^{n(x)}(u(x)>0, u(x) \not \equiv 1)$ 的函数(通常称为幂指函数), 如果

$$
\lim u(x)=a>0, \lim v(x)=b,
$$

那么

$$
\lim u(x)^{\prime \prime(r)}=a^{\prime \prime} .
$$

注意: 这里三个 $\lim$ 都表示在同一自变量变化过程中的极限.

## 习 题 1-9

1. 求函数 $f(x)=\frac{x^{3}+3 x^{2}-x-3}{x^{2}+x-6}$ 的连续区间, 并求极限 $\lim _{x \rightarrow 11} f(x), \lim _{x \rightarrow-3} f(x)$ 及 $\lim _{x \rightarrow 2} f(x)$.
2. 设函数 $f(x)$ 与 $g(x)$ 在点 $x_{0}$ 连续, 证明函数

$$
\varphi(x)=\max \left\{\int(x), g(x) \mid, \quad \psi(x)=\min \{f(x), g(x)\}\right.
$$

在点 $x_{11}$ 也连续.

3. 求下列极限: (1) $\lim _{x \rightarrow 01} \sqrt{x^{2}-2 x+5}$;

(3) $\lim _{x \rightarrow \frac{\pi}{\pi}} \ln (2 \cos 2 x)$; $x \rightarrow \frac{\pi}{6}$

(5) $\lim _{x \rightarrow 1} \frac{\sqrt{5 x-4}-\sqrt{x}}{x-1}$;

(7) $\lim _{x \rightarrow+\infty}\left(\sqrt{x^{2}+x}-\sqrt{x^{2}-x}\right)$.

4. 求下列极限:

(1) $\lim _{x \rightarrow \infty} \mathrm{e}^{\frac{1}{x^{x}}}$;

(2) $\lim _{x \rightarrow 0} \ln \frac{\sin x}{x}$;

(3) $\lim _{1 \rightarrow \infty}\left(1+\frac{1}{x}\right)^{\frac{x}{2}}$;

(4) $\lim _{x \rightarrow 0}\left(1+3 \tan ^{2} x\right)^{a u^{2} \cdot x}$.

(5) $\lim _{x \rightarrow \infty}\left(\frac{3+x}{6+x}\right)^{\frac{r-1}{2}}$;
(2) $\lim _{a \rightarrow \frac{\pi}{4}}(\sin 2 \alpha)^{3}$;

(4) $\lim _{x \rightarrow 1} \frac{\sqrt{x+1}-1}{x}$;

(6) $\lim _{x \rightarrow a} \frac{\sin x-\sin \alpha}{x-\alpha}$;

5. 设 $f(x)$ 在 $\mathbf{R}$ 上连续，且 $f(x) \neq 0, \varphi(x)$ 在 $\mathbf{R}$ 上有定义, 且有间断点，则下列陈述中哪 些是对的，哪些是错的? 如果是对的, 说明理由; 如果是错的，试给出一个反例.

(1) $\varphi[f(x)]$ 必有间断点;

(2) $[\varphi(x)]^{2}$ 必有间断点;

(3) $f[\varphi(x)]$ 未必有间断点;

(6) $\lim _{x \rightarrow 0} \frac{\sqrt{1+\tan x}-\sqrt{1+\sin x}}{x \sqrt{1+\sin ^{2} x}-x}$.

6. 设函数

(4) $\frac{\varphi(x)}{f(x)}$ 必有间断点.

$$
f(x)= \begin{cases}\mathrm{e}^{x}, & x<0, \\ a+x, & x \geqslant 0 .\end{cases}
$$

应当怎样选择数 $a$, 使得 $f(x)$ 成为在 $(-\infty,+\infty)$ 内的连续函数.

## 第十节 闭区间上连续函数的性质

第八节中已说明了函数在区间上连续的概念, 如果函数 $f(x)$ 在开区间 $(a$, $b)$ 内连续, 在右端点 $b$ 左连续, 在左端点 $a$ 右连续, 那么函数 $f(x)$ 就是在闭区 间 $[a, b]$ 上连续的. 在闭区间上连续的函数有几个重要的性质, 今以定理的形式 叙述它们.

## 一、有界性与最大值最小值定理

先说明最大值和最小值的概念. 对于在区间 $I$ 上有定义的函数 $f(x)$, 如果 有 $x_{0} \in I$, 使得对于任一 $x \in I$ 都有

$$
f(x) \leqslant f\left(x_{11}\right) \quad\left(f(x) \geqslant f\left(x_{0}\right)\right),
$$

则称 $f\left(x_{0}\right)$ 是函数 $f(x)$ 在区间 $I$ 上的最大值 (最小值). 例如, 函数 $f(x)=1+\sin x$ 在区间 $[0,2 \pi]$ 上有最大值 2 和最小值 0 . 又例 如, 函数 $f(x)=\operatorname{sgn} x$ 在区间 $(-\infty,+\infty)$ 内有最大值 1 和最小值 -1 . 在开区 间 $(0,+\infty)$ 内, $\operatorname{sgn} x$ 的最大值和最小值都等于 1 (注意: 最大值和最小值可以 相等!). 但函数 $f(x)=x$ 在开区间 $(a, b)$ 内既无最大值又无最小值.下面的定 理给出函数有界且最大值和最小值存在的充分条件.

定理 1(有界性与最大值最小值定理) 在闭区间上连续的函数在该区间上 有界且一定能取得它的最大值和最小值.

这就是说, 如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续, 那么存在常数 $M>0$, 使 得对任一 $x \in[a, b]$, 满足 $|f(x)| \leqslant M$; 且至少有一 点 $\xi_{1}$, 使 $f\left(\xi_{1}\right)$ 是 $f(x)$ 在 $[a, b]$ 上的最大值; 又至少 有一点 $\xi_{2}$, 使 $f\left(\xi_{2}\right)$ 是 $f(x)$ 在 $[a, b]$ 上的最小值 (图 $1-42)$.

这里不予证明.

注意 如果函数在开区间内连续,或函数在闭区 间上有间断点, 那么函数在该区间上不一定有界, 也

$$
y=f(x)= \begin{cases}-x+1, & 0 \leqslant x<1, \\ 1, & x=1, \\ -x+3, & 1<x \leqslant 2\end{cases}
$$

在闭区间 $[0,2]$ 上有间断点 $x=1$, 这函数 $f(x)$ 在闭区 间 $[0,2]$ 上虽然有界, 但是既无最大值又无最小值 (图 1 $-43)$.

因 $1-43$

## 二、零点定理与介值定理

如果 $x_{10}$ 使 $f\left(x_{11}\right)=0$, 则 $x_{0}$ 称为函数 $f(x)$ 的零点.

定理 2(零点定理) 设函数 $f(x)$ 在闭区间 $[a, b]$ 上连续, 且 $f(a)$ 与 $f(b)$ 异号(即 $f(a) \cdot f(b)<0)$, 那么在开区间 $(a, b)$ 内至少有一点 $\xi$, 使

$$
f(\xi)=0 \text {. }
$$

这里不予证明.

从几何上看, 定理 2 表示: 如果连续曲线弧 $y=f(x)$ 的两个端点位于 $x$ 轴 的不同侧,那么这段曲线弧与 $x$ 轴至少有一个交点 (图 1-44). 由定理 2 立即可推得下列较一般性的定理.

定理 3(介值定理) 设函数 $f(x)$ 在闭区间 $[a, b]$ 上连续,且在这区间的端点取不同的函数值

$$
f(a)=A \text { 及 } f(b)=B,
$$

那么,对于 $A$ 与 $B$ 之间的任意一个数 $C$,在开区间 $(a, b)$ 内至少有一点 $\xi$, 使得

$$
f(\xi)=C \quad(a<\xi<b) .
$$

证 设 $\varphi(x)=f(x)-C$, 则 $\varphi(x)$ 在闭区间

$$
\varphi(\xi)=0 \quad(a<\xi<b) .
$$

又 $\varphi(\xi)=f(\xi)-C$, 因此由上式即得

$$
f(\xi)=C \quad(a<\xi<b) .
$$

这定理的几何意义是: 连续曲线弧 $y=f(x)$ 与水平直线 $y=C$ 至少相交于 一点(图 1-45).

推论在闭区间上连续的函数必取得介于最大 值 $M$ 与最小值 $m$ 之间的任何值.

设 $m=f\left(x_{1}\right), M=f\left(x_{2}\right)$, 而 $m \neq M$, 在闭区 间 $\left[x_{1}, x_{2}\right]$ (或 $\left[x_{2}, x_{1}\right]$ ) 上应用介值定理, 即得上 述推论.

例 1 证明方程 $x^{3}-4 x^{2}+1=0$ 在区间 $(0,1)$

证 函数 $f(x)=x^{3}-4 x^{2}+1$ 在闭区间 $[0,1]$ 上连续, 又

$$
f(0)=1>0, f(1)=-2<0 .
$$

根据零点定理, 在 $(0,1)$ 内至少有一点 $\xi$, 使得

$$
f(\xi)=0 \text {, }
$$

即

$$
\xi^{3}-4 \xi^{2}+1=0 \quad(0<\xi<1) .
$$

这等式说明方程 $x^{3}-4 x^{2}+1=0$ 在区间 $(0,1)$ 内至少有一个根是 $\xi$.

## *三、一致连续性

我们先介绍函数的一致连续性概念.

设函数在区间 $I$ 上连续, $x_{0}$ 是在 $I$ 上任意取定的一个点. 由于 $f(x)$ 在点 $x_{0}$ 连续, 因此 $\forall \varepsilon>0, \exists \delta>0$, 使得当 $\left|x-x_{0}\right|<\delta$ 时, 就有 $\left|f(x)-f\left(x_{0}\right)\right|<\varepsilon$. 通常这个 $\delta$ 不仅与 $\varepsilon$ 有关, 而且与所取定的 $x_{0}$ 有关, 即使 $\varepsilon$ 不变, 但选取区间 $I$ 上的其他点作为 $x_{\mathrm{i}}$ 时, 这个 $\delta$ 就不一定适用了. 可是对于某些函数, 却有这样一 种重要情形: 存在着只与 $\varepsilon$ 有关, 而对区间 $I$ 上任何点 $x_{0}$ 都能适用的正数 $\delta$, 即 对任何 $x_{0} \in I$, 只要 $\left|x-x_{0}\right|<\delta$ 时, 就有 $\left|f^{\prime}(x)-f\left(x_{0}\right)\right|<\varepsilon$. 如果函数 $f(x)$ 在区间 $I$ 上能使这种情形发生, 就说函数 $f(x)$ 在区间 $I$ 上是二致连续的.

定义 设函数 $f(x)$ 在区间 $I$ 上有定义. 如果对于任意给定的正数 $\varepsilon$, 总存 在着正数 $\delta$,使得对于区间 $I$ 上的任意两点 $x_{1} 、 x_{2}$, 当 $\left|x_{1}-x_{2}\right|<\delta$ 时,就有

$$
\left|f\left(x_{1}\right)-f\left(x_{2}\right)\right|<\varepsilon,
$$

那么称函数 $f(x)$ 在区间 $I$ 上是一致连续的.

一致连续性表示, 不论在区间 $I$ 的任何部分, 只要自变量的两个数值接近 到一定程度,就可使对应的函数值达到所指定的接近程度.

由上述定义可知, 如果函数 $f(x)$ 在区间 $I$ 上一致连续,那么 $f(x)$ 在区间 $I$ 上也是连续的.但反过来不一定成立,举例说明如下:

例 2 函数 $f(x)=\frac{1}{x}$ 在区间 $(0,1]$ 上是连续的, 但不是一致连续的.

因为函数 $f(x)=\frac{1}{x}$ 是初等函数, 它在区间 $(0,1]$ 上有定义, 所以在 $(0,1]$ 上 是连续的.

$\forall \varepsilon>0(0<\varepsilon<1)$, 假定 $f(x)=\frac{1}{x}$ 在 $(0,1]$ 上一致连续, 应该 $\exists \delta>0$, 使得 对于 $(0,1]$ 上的任意两个值 $x_{1}, x_{2}$, 当 $\left|x_{1}-x_{2}\right|<\delta$ 时, 就有 $\left|f\left(x_{1}\right)-f\left(x_{2}\right)\right|$ $<\varepsilon$.

现在取原点附近的两点

$$
x_{1}=\frac{1}{n}, \quad x_{2}=\frac{1}{n+1},
$$

其中 $n$ 为正整数,这样的 $x_{1} 、 x_{2}$ 显然在 $(0,1]$ 上. 因

$$
\left|x_{1}-x_{2}\right|=\left|\frac{1}{n}-\frac{1}{n+1}\right|=\frac{1}{n(n+1)},
$$

故只要 $n$ 取得足够大, 总能使 $\left|x_{1}-x_{2}\right|<\delta$. 但这时有

$$
\left|f\left(x_{1}\right)-f\left(x_{2}\right)\right|=\left|\frac{1}{\frac{1}{n}}-\frac{1}{\frac{1}{n+1}}\right|=|n-(n+1)|=1>\varepsilon,
$$

不符合一致连续的定义, 所以 $f(x)=\frac{1}{x}$ 在 $(0,1]$ 上不是一致连续的.

上例说明,在半开区间上连续的函数不一定在该区间上一致连续. 但是, 有下面的定理：

定理 4(一致连续性定理) 如果函数 $f(x)$ 在闭区间 $[a, b]$ 上连续, 那么它在该区间上一致连续。

这里不予证明。

## 习 题 $\quad 1-10$

1. 假设函数 $f(x)$ 在闭区间 $[0,1]$ 上连续,并且对 $[0,1]$ 上任一点 $x$ 有 $0 \leqslant f(x) \leqslant 1$. 试证 明 $[0,1]$ 中必存在一点 $c$, 使得 $f^{\prime}(c)=c$ ( $c$ 称为函数 $f(x)$ 的不动点).
2. 证明方程 $x^{5}-3 x=1$ 至少有一个根介于 1 和 2 之间.
3. 证明方程 $x=a \sin x+b$, 其中 $a>0, b>0$, 至少有一个正根, 并且它不超过 $a+b$.

-4. 设函数 $f(x)$ 对于闭区间 $[a, b]$ 上的任意两点 $x 、 y$, 恒有 $\mid f(x)-$ $f(y)|\leqslant L| x-y \mid$, 其中 $L$ 为正常数, 且 $f^{\prime}(a) \cdot f(b)<0$. 证明: 至少有一点 $\xi \in(a, b)$, 使得 $f(\xi)=0$.

5. 若 $f(x)$ 在 $[a, b]$ 上连续, $a<x_{1}<x_{2}<\cdots<x_{n}<b(n \geqslant 3)$, 则在 $\left(x_{1}, x_{n}\right)$ 内至少有一 点 $\xi$.使 $f(\xi)=\frac{f\left(x_{1}\right)+f\left(x_{2}\right)+\cdots+f\left(x_{n}\right)}{n}$.
6. 证明: 若 $f(x)$ 在 $(-\infty,+\infty)$ 内连续, 且 $\lim _{x \rightarrow \infty} f(x)$ 存在, 则 $f(x)$ 必在 $(-\infty,+\infty)$ 内 有界.
7. 在什么条件下, $(a, b)$ 内的连续函数 $f(x)$ 为一致连续?

## 总习题一

1. 在“充分”、“必要”和“充分必要”三者中选择一个正确的填入下列空格内：

(1) 数列 $\left\{x_{n}\right\}$ 有界是数列 $\left\{x_{n}\right\}$ 收敛的 条件.数列 $\left|x_{n}\right|$ 收敛是数列 $\left\{x_{n}\right\}$ 有界的 条件.

(2) $f(x)$ 在 $x_{0}$ 的某一去心邻域内有界是 $\lim _{x \rightarrow x_{0}} f(x)$ 存在的 条件. $\lim _{x \rightarrow x_{0}} f(x)$ 存在 是 $f(x)$ 在 $x_{0}$ 的某一去心邻域内有界的 条件.

(3) $f(x)$ 在 $x_{0}$ 的某一去心邻域内无界是 $\lim _{\text {. }} f(x)=\infty$ 的 条件. $\lim _{x \rightarrow r_{0}} f(x)=\infty$ 是 $f(x)$ 在 $x_{0}$ 的某一去心邻域内无界的 条件.

(4) $f(x)$ 当 $x \rightarrow x_{0}$ 时的右极限 $f\left(x_{0}^{+}\right)$及左极限 $f\left(x_{0}^{-}\right)$都存在且相等是 $\lim _{x \rightarrow x_{0}} f(x)$ 存在 的 条件。

2. 已知函数

$$
f(x)=\left\{\begin{array}{cc}
(\cos x)^{-x^{2}}, & x \neq 0, \\
a, & x=0
\end{array}\right.
$$

在 $x=0$ 连续, 则 $a=$

3. 选择以下两题中给出的四个结论中一个正确的结论.

(1) 设 $f(x)=2^{x}+3^{x}-2$. 则当 $x \rightarrow 0$ 时, 有 $(\quad)$.
(A) $f(x)$ 与 $x$ 是等价无穷小.
(B) $f(x)$ 与 $x$ 同阶但非等价无穷小.
(C) $f(x)$ 是比 $x$ 高阶的无穷小.
(D) $f(x)$ 是比 $x$ 低阶的无穷小.
(2) 设

$$
f(x)=\frac{\mathrm{e}^{\frac{1}{x}}-1}{\mathrm{e}^{\frac{1}{x}}+1},
$$

则 $x=0$ 是 $f(x)$ 的 ( )
(A) 可去间断点.
(B) 跳跃间断点.
(C) 第二类间断点.
(D) 连续点.

4. 设 $f(x)$ 的定义域是 $[0,1]$, 求下列函数的定义域:
(1) $f\left(\mathrm{e}^{x}\right)$;
(2) $f(\ln x)$;
(3) $f(\arctan x)$;
(4) $f(\cos x)$.
5. 设

$$
f(x)=\left\{\begin{array}{ll}
0, & x \leqslant 0, \\
x, & x>0,
\end{array} g(x)= \begin{cases}0, & x \leqslant 0, \\
-x^{2}, & x>0,\end{cases}\right.
$$

求 $f[f(x)], g[g(x)], f[g(x)], g[f(x)]$.

6. 利用 $y=\sin x$ 的图形作出下列函数的图形:
(1) $y=|\sin x|$;
(2) $y=\sin |x|$;
(3) $y=2 \sin \frac{x}{2}$.
7. 把半径为 $R$ 的一圆形铁皮,自中心处音去中心角为 $a$ 的一扇形后围成一无底图锥. 试将这圆锥的体积表为 $a$ 的函数.
8. 根据函数极限的定义证明 $\lim _{x \rightarrow 3} \frac{x^{2}-x-6}{x-3}=5$.
9. 求下列极限:
(1) $\lim _{x \rightarrow 1} \frac{x^{2}-x+1}{(x-1)^{2}}$;
(2) $\lim _{x \rightarrow+\infty} x\left(\sqrt{x^{2}+1}-x\right)$;
(3) $\lim _{x \rightarrow \infty}\left(\frac{2 x+3}{2 x+1}\right)^{x+1}$;
(4) $\lim _{x \rightarrow 0} \frac{\tan x-\sin x}{x^{3}}$;
(5) $\lim _{x \rightarrow 0}\left(\frac{a^{x}+b^{x}+c^{x}}{3}\right)^{\frac{1}{x}} \quad(a>0, b>0, c>0)$;
(6) $\lim _{x \rightarrow \frac{\pi}{x}}(\sin x)^{\operatorname{lan} x}$.
10. 设

$$
f(x)= \begin{cases}x \sin \frac{1}{x}, & x>0, \\ a+x^{2} & x \leqslant 0,\end{cases}
$$

要使 $f(x)$ 在 $(-\infty,+\infty)$ 内连续, 应当怎样选择数 $a$ ?

11. 设

$$
f(x)= \begin{cases}\mathrm{e}^{\frac{1}{x-7}}, & x>0, \\ \ln (1+x), & -1<x \leqslant 0,\end{cases}
$$

求 $f(x)$ 的间断点。并说明间断点所属类型。

12. 证明

$$
\lim _{n \rightarrow \infty}\left(\frac{1}{\sqrt{n^{2}+1}}+\frac{1}{\sqrt{n^{2}+2}}+\cdots+\frac{1}{\sqrt{n^{2}+n}}\right)=1
$$

13. 证明方程 $\sin x+x+1=0$ 在开区间 $\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ 内至少有一个根.
14. 如果存在直线 $L: y=k x+b$, 使得当 $x \rightarrow \infty$ (或 $x \rightarrow+\infty, x \rightarrow-\infty$ ) 时, 曲线 $y=f(x)$ 上的动点 $M(x, y)$ 到直线 $L$ 的距离 $d(M, L) \rightarrow 0$, 则称 $L$ 为曲线 $y=f(x)$ 的海近线. 当直线 $L$ 的斜率 $k \neq 0$ 时, 称 $L$ 为斜浙近线.

(1) 证明:直线 $L: y=k x+6$ 为曲线 $y=f(x)$ 的渐近线的充分必要条件是

(2) 求曲线 $y=(2 x-1) \mathrm{e}^{\frac{1}{x}}$ 的斜渐近线.

$$
k=\lim _{\substack{x \rightarrow \infty \\ x \rightarrow+\infty \\ x \rightarrow-\infty}} \frac{f(x)}{x}, \quad b=\lim _{\substack{x \rightarrow \infty \\ r \rightarrow+\infty \\ x \rightarrow-\infty}}[f(x)-k x]
$$

(2) 求曲线 $y=(2 x-1) \mathrm{e}^{\frac{1}{x}}$ 的斜渐近线.

## 第二章 导数与微分

微分学是微积分的重要组成部分,它的基本概念是导数与微分.

本章中, 我们主要讨论导数和微分的概念以及它们的计算方法. 至于导数的 应用,将在第三章讨论.

## 第一节 导 数 概 念

## 一、引例

为了说明微分学的基本概念一一导数,我们先讨论两个问题: 速度问题和切 线问题. 这两个问题在历史上都与导数概念的形成有密切的关系.

## 1. 直线运动的速度

设某点沿直线运动. 在直线上引人原点和单位点 (即表示实数 1 的点), 使直 线成为数轴. 此外, 再取定一个时刻作为测墨时间的零点. 设动点于时刻 $t$ 在直 线上的位置的坐标为 $s$ (简称位置 $s$ ). 这样, 该点的运动完全由某个函数

$$
s=f(t)
$$

所确定. 此函数对运动过程中所出现的 $\iota$ 值有定义, 称为位置函数. 在最简单的 情形,该动点所经过的路程与所花的时间成正比. 就是说, 无论取哪一段时间间 隔, 比值

$$
\frac{\text { 经过的路程 }}{\text { 所花的时间 }}
$$

总是相同的. 这个比值就称为该动点的速度, 并说该点作包速运动. 如果运动不 是匀速的, 那么在运动的不同时间间隔内, 比值 (1) 会有不同的值. 这样, 把比值 (1)笼统地称为该动点的速度就不合适了, 而需要按不同时刻来考虑. 那么, 这种 非匀速运动的动点在某一时刻 (设为 $t_{0}$ ) 的速度应如何理解而又如何求得呢?

首先取从时刻 $t_{0}$ 到 $t$ 这样一个时间间隔, 在这段时间内, 动点从位置 $s_{0}=$ $f\left(t_{0}\right)$ 移动到 $s=f(t)$. 这时由 (1) 式算得的比值

$$
\frac{s-s_{0}}{t-t_{0}}=\frac{f(t)-f\left(t_{0}\right)}{t-t_{0}}
$$

可认为是动点在上述时间间隔内的平均速度. 如果时间间隔选得较短, 这个比值 (2) 在实践中也可用来说明动点在时刻 $t_{0}$ 的速度. 但对于动点在时刻 $t_{0}$ 的速度 的精确概念来说, 这样做是不够的, 而更确切地应当这样: 令 $t \rightarrow t_{0}$, 取 (2) 式的 极限, 如果这个极限存在; 设为 $v$, 即

$$
v=\lim _{t \rightarrow t_{0}} \frac{f(t)-f\left(t_{0}\right)}{t-t_{0}},
$$

这时就把这个极限值 $v$ 称为动点在时刻 $\iota_{0}$ 的 (瞬时)速度.

## 2. 切线问题

圆的切线可定义为“与曲线只有一个交点的直线”. 但是对于其他曲线, 用 “与曲线只有一个交点的直线”作为切线的定义就不一定合适. 例如, 对于扡物线 $y=x^{2}$, 在原点 $O$ 处两个坐标轴都符合上述定义, 但实际上只有 $x$ 轴是该抛物 线在点 $O$ 处的切线.下面给出切线的定义.

设有曲线 $C$ 及 $C$ 上的一点 $M$ (图 2-1), 在点 $M$ 外另取 $C$ 上一点 $N$, 作割 线 $M N$. 当点 $N$ 沿曲线 $C$ 趋于点 $M$ 时, 如果割线 $M N$ 绕点 $M$ 旋转而趋于极限 位置 $M T$, 直线 $M T$ 就称为曲线 $C$ 在点 $M$ 处的切线. 这里极限位置的含义是: 只 要弦长 $|M N|$ 趋于零, $\angle N M T$ 也趋于零.

现在就曲线 $C$ 为函数 $y=f(x)$ 的图形的情形来讨论切线问题. 设 $M\left(x_{01}\right.$, $y_{0}$ ) 是曲线 $C$ 上的一个点 (图 2-2), 则 $y_{10}=f\left(x_{0}\right)$. 根据上述定义要定出曲线 $C$ 在点 $M$ 处的切线, 只要定出切线的斜率就行了. 为此, 在点 $M$ 外另取 $C$ 上的一 点 $N(x, y)$,于是割线 $M N$ 的斜率为

$$
\tan \varphi=\frac{y-y_{10}}{x-x_{0}}=\frac{f(x)-f\left(x_{01}\right)}{x-x_{0}},
$$

其中 $\varphi$ 为制线 $M N$ 的倾角. 当点 $N$ 沿曲线 $C$ 趋于点 $M$ 时, $x \rightarrow x_{0}$. 如果当 $x \rightarrow$ $x_{0}$ 时, 上式的极限存在, 设为 $k$, 即

$$
k=\lim _{x \rightarrow=x_{0}} \frac{f(x)-f\left(x_{0}\right)}{x-x_{0}}
$$

存在, 则此极限 $k$ 是割线斜率的极限, 也就是切线的斜率. 这里 $k=\tan \alpha$, 其中 $\alpha$ 是切线 $M T$ 的倾角. 于是, 通过点 $M\left(x_{0}, f\left(x_{0}\right)\right)$ 且以 $k$ 为斜率的直线 $M T$ 便是 曲线 $C$ 在点 $M$ 处的切线. 事实上, 由 $\angle N M T=\varphi-\alpha$ 以及 $x \rightarrow x_{0}$ 时 $\varphi \rightarrow \alpha$, 可见 $x \rightarrow x_{0}$ 时 (这时 $|M N| \rightarrow 0$ ), $\angle N M T \rightarrow 0$. 因此直线 $M T$ 确为曲线 $C$ 在点 $M$ 处 的切线.

## 二、导数的定义

## 1. 函数在一点处的导数与导函数

从上面所讨论的两个问题看出，非匀速直线运动的速度和切线的斜率都归 结为如下的极限：

$$
\lim _{x \rightarrow x_{0}} \frac{f(x)-f\left(x_{0}\right)}{x-x_{0}}
$$

这里 $x-x_{10}$ 和 $f(x)-f\left(x_{11}\right)$ 分别是函数 $y=f(x)$ 的自变量的增量 $\Delta x$ 和函数 的增量 $\Delta y$ :

$$
\begin{gathered}
\Delta x=x-x_{0}, \\
\Delta y=f(x)-f\left(x_{0}\right)=f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right) .
\end{gathered}
$$

因 $x \rightarrow x_{0}$ 相当于 $\Delta x \rightarrow 0$, 故 (3)式也可写成

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x} \text { 或 } \lim _{\Delta x \rightarrow 0} \frac{f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)}{\Delta x} \text {. }
$$

在自然科学和工程技术领域内, 还有许多概念, 例如电流强度、角速度、线密度等 等, 都可归结为形如 (3)式的数学形式. 我们撤开这些量的具体意义,抓住它们在 数量关系上的共性, 就得出函数的导数概念.

定义 设函数 $y=f(x)$ 在点 $x_{0}$ 的某个邻域内有定义, 当自变量 $x$ 在 $x_{0}$ 处 取得增量 $\Delta x$ (点 $x_{0}+\Delta x$ 仍在该邻域内) 时, 相应的函数取得增墨 $\Delta y=f\left(x_{0}+\right.$ $\Delta x)-f\left(x_{0}\right)$; 如果 $\Delta y$ 与 $\Delta x$ 之比当 $\Delta x \rightarrow 0$ 时的极限存在, 则称函数 $y=$ $f(x)$ 在点 $x_{0}$ 处可导, 并称这个极限为函数 $y=f(x)$ 在点 $x_{0}$ 处的导数, 记为 $f^{\prime}\left(x_{0}\right)$, 即

$$
f^{\prime}\left(x_{0}\right)=\lim _{\Delta, r \rightarrow 0} \frac{\Delta y}{\Delta x}=\lim _{\Delta x \rightarrow 0} \frac{f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)}{\Delta x},
$$

也可记作 $\left.y^{\prime}\right|_{x=x_{0}},\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{x=x_{0}}$ 或 $\left.\frac{\mathrm{d} f(x)}{\mathrm{d} x}\right|_{x=x_{10}}$.

函数 $f(x)$ 在点 $x_{0}$ 处可导有时也说成 $f(x)$ 在点 $x_{0}$ 具有导数或导数存在.

导数的定义式(4)也可取不同的形式,常见的有

$$
f^{\prime}\left(x_{0}\right)=\lim _{h \rightarrow 0} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h}
$$

$$
f^{\prime}\left(x_{0}\right)=\lim _{x \rightarrow-x_{0}} \frac{f(x)-f\left(x_{0}\right)}{x-x_{0}} .
$$

(5)式中的 $h$ 即自变量的增量 $\Delta x$.

在实际中,需要讨论各种具有不同意义的变量的变化“快慢”问题,在数学上 就是所调函数的恋化率问题. 导数概念就是函数变化率这一概念的精确描述. 它 做开了自变量和因变量所代表的几何或物理等方面的特殊意义,纯粹从数量方 面来刻画变化率的本质: 因变量增量与自变量增量之比 $\frac{\Delta y}{\Delta x}$ 是因变量 $y$ 在以 $x_{0}$ 和 $x_{0}+\Delta x$ 为端点的区间上的平均变化率, 而导数 $f^{\prime}\left(x_{0}\right)$ 则是因变量 $y$ 在点 $x_{0}$ 处的变化率,它反映了因变量随自变量的变化而变化的快慢程度.

如果极限 (4) 不存在, 就说函数 $y=f(x)$ 在点 $x_{0}$ 处不可导. 如果不可导的 原因是由于 $\Delta x \rightarrow 0$ 时, 比式 $\frac{\Delta y}{\Delta x} \rightarrow \infty$, 为了方便起见, 也往往说函数 $y=f(x)$ 在 点 $x_{0}$ 处的导数为无穷大.

上面讲的是函数在一点处可导.如果函数 $y=f(x)$ 在开区间 $I$ 内的每点处 都可导, 就称函数 $f(x)$ 在开区间 $I$ 内可导. 这时, 对于任一 $x \in I$, 都对应着 $f^{\prime}(x)$ 的一个确定的导数值. 这样就构成了一个新的函数, 这个函数叫做原来函 数 $y=f(x)$ 的导函数,记作 $y^{\prime}, f^{\prime}(x), \frac{\mathrm{d} y}{\mathrm{~d} x}$ 或 $\frac{\mathrm{d} f(x)}{\mathrm{d} x}$.

在 (4) 式或 (5)式中把 $x_{\mathrm{v}}$ 换成 $x$, 即得导函数的定义式

$$
y^{\prime}=\lim _{\Delta x \rightarrow 0} \frac{f(x+\Delta x)-f(x)}{\Delta x}
$$

或

$$
f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h} .
$$

注意 在以上两式中,虽然 $x$ 可以取区间 $I$ 内的任何数值,但在极限过程 中, $x$ 是常照, $\Delta x$ 或 $h$ 是变罢.

显然, 函数 $f(x)$ 在点 $x_{10}$ 处的导数 $f^{\prime}\left(x_{0}\right)$ 就是导函数 $f^{\prime}(x)$ 在点 $x=x_{0}$ 处 的函数值, 即

$$
f^{\prime}\left(x_{\mathrm{n}}\right)=\left.f^{\prime}(x)\right|_{, \mathrm{r}=x_{\mathrm{n}}} .
$$

导函数 $f^{\prime}(x)$ 简称导数, 而 $f^{\prime}\left(x_{0}\right)$ 是 $f(x)$ 在 $x_{0}$ 处的导数或导数 $f^{\prime \prime}(x)$ 在 $x_{0}$ 处的值.

## 2. 求导数举例

下面根据导数定义求一些简单函数的导数. 例 1 求函数 $f(x)=C$ ( $C$ 为常数) 的导数.

解 $f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}=\lim _{h \rightarrow 0} \frac{C-C}{h}=0$,

即

$$
(C)^{\prime}=0 \text {. }
$$

这就是说, 常数的导数等于零.

例 2 求函数 $f(x)=x^{n} \quad\left(n \in \mathbf{N}^{+}\right)$在 $x=a$ 处的导数.

解 $f^{\prime}(a)=\lim _{x \rightarrow a} \frac{f(x)-f(a)}{x-a}=\lim _{x \rightarrow a} \frac{x^{n}-a^{\prime \prime}}{x-a}$

$$
=\lim _{x \rightarrow u}\left(x^{n-1}+a x^{n-2}+\cdots+a^{n-1}\right)=n a^{n-1} .
$$

把以上结果中的 $a$ 换成 $x$ 得 $f^{\prime}(x)=n x^{n-1}$, 即

$$
\left(x^{\prime \prime}\right)^{\prime}=n x^{n-1} \text {. }
$$

更一般的,对于幂函数 $y=x^{\prime \prime}$ ( $\mu$ 为常数), 有

$$
\left(x^{\mu}\right)^{\prime}=\mu x^{\mu-1} \text {. }
$$

这就是幂函数的导数公式. 这公式的证明将在以后讨论. 利用这公式, 可以很方 便地求出幂函数的导数,例如:

当 $\mu=\frac{1}{2}$ 时, $y=x^{\frac{1}{2}}=\sqrt{x}(x>0)$ 的导数为

即

$$
\left(x^{\frac{1}{2}}\right)^{\prime}=\frac{1}{2} x^{\frac{1}{2}-1}=\frac{1}{2} x^{-\frac{1}{2}},
$$

即

$$
(\sqrt{x})^{\prime}=\frac{1}{2 \sqrt{x}}
$$

当 $\mu=-1$ 时, $y=x^{-1}=\frac{1}{x}(x \neq 0)$ 的导数为

$$
\left(x^{-1}\right)^{\prime}=(-1) x^{-1-1}=-x^{-2},
$$

$$
\left(\frac{1}{x}\right)^{\prime}=-\frac{1}{x^{2}} \text {. }
$$

例 3 求函数 $f(x)=\sin x$ 的导数.

解 $f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}=\lim _{h \rightarrow 0} \frac{\sin (x+h)-\sin x}{h}$

$$
\begin{aligned}
& =\lim _{h \rightarrow 0} \frac{1}{h} \cdot 2 \cos \left(x+\frac{h}{2}\right) \sin \frac{h}{2} \\
& =\lim _{h \rightarrow-1)} \cos \left(x+\frac{h}{2}\right) \cdot \frac{\sin \frac{h}{2}}{\frac{h}{2}}=\cos x,
\end{aligned}
$$

即

$$
(\sin x)^{\prime}=\cos x .
$$

这就是说,正弦函数的导数是余弦函数. 用类似的方法, 可求得

$$
(\cos x)^{\prime}=-\sin x,
$$

就是说, 余弦函数的导数是负的正弦函数.

例 4 求函数 $f(x)=a^{x}(a>0, a \neq 1)$ 的导数.

解 $f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}=\lim _{h \rightarrow 0} \frac{a^{a^{x+h}}-a^{x}}{h}$

$$
=a^{x} \lim _{h \rightarrow 0} \frac{a^{h}-1}{h} .
$$

利用第一章第九节例 7 的结果得

即

$$
f^{\prime}(x)=a^{x} \ln a,
$$

$$
\left(a^{x}\right)^{\prime}=a^{x} \ln a .
$$

这就是指数函数的导数公式. 特殊地, 当 $a=\mathrm{e}$ 时,因 $\ln \mathrm{e}=1$, 故有

$$
\left(e^{x}\right)^{\prime}=e^{x} \text {. }
$$

上式表明, 以 $\mathrm{e}$ 为底的指数函数的导数就是它自己,这是以 $\mathrm{e}$ 为底的指数函 数的一个重要特性.

例 5 求函数 $f(x)=\log _{a} x(a>0, a \neq 1)$ 的导数.

解 $f^{\prime}(x)=\lim _{h \rightarrow 0} \frac{f(x+h)-f(x)}{h}=\lim _{h \rightarrow 0} \frac{\log _{a}(x+h)-\log _{a} x}{h}$

$$
=\lim _{h \rightarrow 11} \frac{1}{h} \log _{a} \frac{x+h}{x}=\lim _{h \rightarrow 0} \frac{1}{x} \cdot \frac{x}{h} \log _{a}\left(1+\frac{h}{x}\right)
$$

$$
=\frac{1}{x} \lim _{h \rightarrow 0} \frac{\log _{a}\left(1+\frac{h}{x}\right)}{\frac{h}{x}} \text {. }
$$

作代换 $u=\frac{h}{x}$ 并利用第一章第九节例 6 的结果得

即

$$
\begin{gathered}
f^{\prime}(x)=\frac{1}{x \ln a} \\
\left(\log _{a} x\right)^{\prime}=\frac{1}{x \ln a} .
\end{gathered}
$$

这就是对数函数的导数公式. 特殊地, 当 $a=\mathrm{e}$ 时, 由上式得自然对数函数 的导数公式

$$
(\ln x)^{\prime}=\frac{1}{x} .
$$

例 6 求函数 $f(x)=|x|$ 在 $x=0$ 处的导数.

解 $\lim _{h \rightarrow 0} \frac{f(0+h)-f(0)}{h}=\lim _{h \rightarrow 0} \frac{|h|-0}{h}=\lim _{h \rightarrow 0} \frac{|h|}{h}$. 当 $h<0$ 时, $\frac{|h|}{h}=-1$, 故 $\lim _{h \rightarrow 0^{-}} \frac{|h|}{h}=-1$;

当 $h>0$ 时, $\frac{|h|}{h}=1$, 故 $\lim _{h \rightarrow 0^{+}} \frac{|h|}{h}=1$.

所以, $\lim _{h \rightarrow 0} \frac{f(0+h)-f(0)}{h}$ 不存在, 即函数 $f(x)=|x|$ 在 $x=0$ 处不可导.

## 3. 单侧导数

根据函数 $f(x)$ 在点 $x_{0}$ 处的导数 $f^{\prime}\left(x_{0}\right)$ 的定义, 导数

$$
f^{\prime}\left(x_{0}\right)=\lim _{h \rightarrow 0} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h}
$$

是一个极限, 而极限存在的充分必要条件是左、右极限都存在且相等, 因此 $f^{\prime}\left(x_{0}\right)$ 存在即 $f(x)$ 在点 $x_{0}$ 处可导的充分必要条件是左、右极限

$$
\lim _{h \rightarrow 0^{-}} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h} \text { 及 } \lim _{h \rightarrow 1^{+}} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h}
$$

都存在且相等. 这两个极限分别称为函数 $f^{\prime}(x)$ 在点 $x_{0}$ 处的左导数和右导数, 记作 $f_{-}^{\prime}\left(x_{0}\right)$ 及 $f_{+}^{\prime}\left(x_{0}\right)$, 即

$$
\begin{aligned}
& f^{\prime}\left(x_{0}\right)=\lim _{h \rightarrow 0^{-}} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h}, \\
& f^{\prime}\left(x_{11}\right)=\lim _{h \rightarrow 0^{+}} \frac{f\left(x_{0}+h\right)-f\left(x_{0}\right)}{h} .
\end{aligned}
$$

现在可以说, 函数 $f(x)$ 在点 $x_{0}$ 处可导的充分必要条件是左导数 $f^{\prime} .\left(x_{0}\right)$ 和右 导数 $f_{+}^{\prime \prime}\left(x_{0}\right)$ 都存在且相等.

函数 $f(x)=|x|$ 在 $x=0$ 处的左导数 $f^{\prime}-(0)=-1$ 及右导数 $f^{\prime},(0)=1$ 虽 然都存在,但不相等, 故 $f(x)=|x|$ 在 $x=0$ 处不可导.

左导数和右导数统称为单侧导数.

如果函数 $f(x)$ 在开区间 $(a, b)$ 内可导, 且 $f^{\prime}(a)$ 及 $f_{-}^{\prime}(b)$ 都存在, 就说 $f(x)$ 在闭区间 $[a, b]$ 上可导.

## 三、导数的几何意义

由第一目中切线问题的讨论以及第二目中 导数的定义可知: 函数 $y=f(x)$ 在点 $x_{0}$ 处的 导数 $f^{\prime}\left(x_{0}\right)$ 在几何上表示曲线 $y=f(x)$ 在点 $M\left(x_{11}, f\left(x_{0}\right)\right)$ 处的切线的斜率, 即

$$
f^{\prime}\left(x_{11}\right)=\tan \alpha,
$$

其中 $\alpha$ 是切线的倾角 (图 2-3).

根据导数的几何意义并应用直线的点斜式方程, 可知曲线 $y=f(x)$ 在点 $M\left(x_{0}, y_{0}\right)$ 处的切线方程为

$$
y-y_{0}=f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right) .
$$

过切点 $M\left(x_{0}, y_{0}\right)$ 且与切线垂直的直线叫做曲线 $y=f(x)$ 在点 $M$ 处的法 线. 如果 $f^{\prime}\left(x_{0}\right) \neq 0$, 法线的斜率为 $-\frac{1}{f^{\prime}\left(x_{0}\right)}$, 从而法线方程为

$$
y-y_{0}=-\frac{1}{f^{\prime}\left(x_{0}\right)}\left(x-x_{0}\right) \text {. }
$$

例 7 求等边双曲线 $y=\frac{1}{x}$ 在点 $\left(\frac{1}{2}, 2\right)$ 处的切线的斜率, 并写出在该点处 的切线方程和法线方程.

解 根据导数的几何意义知道,所求切线的斜率为

$$
k_{1}=\left.y^{\prime}\right|_{x=\frac{1}{2}} \text {. }
$$

由于 $y^{\prime}=\left(\frac{1}{x}\right)^{\prime}=-\frac{1}{x^{2}}$, 于是

$$
k_{1}=-\left.\frac{1}{x^{2}}\right|_{, x=\frac{1}{2}}=-4
$$

从而所求切线方程为

即

$$
\begin{gathered}
y-2=-4\left(x-\frac{1}{2}\right), \\
4 x+y-4=0 .
\end{gathered}
$$

所求法线的斜率为

$$
k_{2}=-\frac{1}{k_{1}}=\frac{1}{4},
$$

于是所求法线方程为

$$
y-2=\frac{1}{4}\left(x-\frac{1}{2}\right),
$$

即

$$
2 x-8 y+15=0 \text {. }
$$

例 8 求曲线 $y=x^{\frac{3}{2}}$ 的通过点 $(0,-4)$ 的切线方程.

解 设切点为 $\left(x_{10}, y_{10}\right)$, 则切线的斜率为

$$
f^{\prime}\left(x_{0}\right)=\left.\frac{3}{2} \sqrt{x}\right|_{, \therefore x_{11}}=\frac{3}{2} \sqrt{x_{0}} .
$$

于是所求切线方程可设为

$$
y-y_{0}=\frac{3}{2} \sqrt{x_{0}}\left(x-x_{0}\right) .
$$

切点 $\left(x_{0}, y_{0}\right)$ 在曲线 $y=x^{\frac{3}{2}}$ 上, 故有

$$
y_{0}=x_{0}^{\frac{3}{2}},
$$

切线 (7) 通过点 $(0,-4)$, 故有

$$
-4-y_{0}=\frac{3}{2} \sqrt{x_{0}}\left(0-x_{i 1}\right) \text {. }
$$

求得方程 (8) 及 (9)组成的方程组的解为 $x_{0}=4, y_{0}=8$, 代入 (7) 式并化简, 即得所求切线方程为

$$
3 x-y-4=0
$$

## 四、函数可导性与连续性的关系

设函数 $y=f(x)$ 在点 $x$ 处可导, 即

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=f^{\prime}(x)
$$

存在. 由具有极限的函数与无穷小的关系知道,

$$
\frac{\Delta y}{\Delta x}=f^{\prime}(x)+\alpha,
$$

其中 $a$ 为当 $\Delta x \rightarrow 0$ 时的无穷小. 上式两边同乘以 $\Delta x$, 得

$$
\Delta y=f^{\prime}(x) \Delta x+\alpha \Delta x \text {. }
$$

由此可见, 当 $\Delta x \rightarrow 0$ 时, $\Delta y \rightarrow 0$. 这就是说, 函数 $y=f(x)$ 在点 $x$ 处是连续的. 所 以, 如果函数 $y=f(x)$ 在点 $x$ 处可导, 则函数在该点必连续.

另一方面,一个函数在某点连续却不一定在该点可导.举例说明如下:

例 9 函数 $y=f(x)=\sqrt[3]{x}$ 在区间 $(-\infty,+\infty)$ 内连续, 但在点 $x=0$ 处不可 导. 这是因为在点 $x=0$ 处有

$$
\frac{f(0+h)-f(0)}{h}=\frac{\sqrt[3]{h}-0}{h}=\frac{1}{h^{2 / 3}},
$$

因而, $\lim _{h \rightarrow 0} \frac{f(0+h)-f(0)}{h}=\lim _{h \rightarrow-11} \frac{1}{h^{2 / 3}}=+\infty$, 即导数为无穷大 (注意, 导数不存 在). 这事实在图形中表现为曲线 $y=\sqrt[3]{x}$ 在原点 $O$ 具有垂直于 $x$ 轴的切线 $x=0$ (图 2-4).

例 10 函数 $y=\sqrt{x^{2}}$ (即 $\left.y=|x|\right)$ 在 $(-\infty,+\infty)$ 内连续, 但在例 6 中已经 看到, 这函数在 $x=0$ 处不可导. 曲线 $y=\sqrt{x^{2}}$ 在原点 $O$ 没有切线(图 2-5).

由以上讨论可知, 函数在某点连续是函数在该点可导的必要条件,但不是充 分条件.

## 习 题 2-1

1. 设物体绕定轴旋转, 在时间间隔 $[0, t]$ 内转过角度 $\theta$, 从而转角 $\theta$ 是 $t$ 的函数: $\theta=$ $\theta(t)$. 如果旋转是匀速的,那么称 $\omega=\frac{\theta}{t}$ 为该物体旋转的角速庶. 如果旋转是非匀速的, 应怎 样确定该物体在时刻 $t_{0}$ 的角速度?
2. 当物体的温度高于周围介质的温度时,物体就不断冷却. 若物体的温度 $T$ 与时间 $t$ 的 函数关系为 $T=T(t)$, 应怎样确定该物体在时刻 $t$ 的冷却速度?
3. 设某工厂生产 $x$ 件产品的成本为

$$
C(x)=2000+100 x-0.1 x^{2}(\text { 元 }) \text {, }
$$

这函数 $C(x)$ 称为成本函数; 成本函数 $C(x)$ 的导数 $C^{\prime}(x)$ 在经济学中称为边际成本. 试求

(1) 当生产 100 件产品时的边际成本;

（2）生产第 101 件产品的成本，并与(1)中求得的边际成本作比较,说明边际成本的实际 意义。

4. 设 $f(x)=10 x^{2}$, 试按定义求 $f^{\prime}(-1)$.
5. 证明 $(\cos x)^{\prime}=-\sin x$.
6. 下列各题中均假定 $f^{\prime}\left(x_{n}\right)$ 存在，按照导数定义观察下列极限，指出 $A$ 表示什么:

(1) $\lim _{\Delta x \rightarrow 0} \frac{f\left(x_{0}-\Delta x\right)-f\left(x_{0}\right)}{\Delta x}=A$;

(2) $\lim _{x \rightarrow 0} \frac{f(x)}{x}=A$, 其中 $f(0)=0$, 且 $f^{\prime}(0)$ 存在;

(3) $\lim _{h \rightarrow-11} \frac{f\left(x_{01}+h\right)-f\left(x_{0}-h\right)}{h}=A$.

以下两题中, 选择给出的四个结论中一个正确的结论: 7. 设

$$
f(x)= \begin{cases}\frac{2}{3} x^{3}, & x \leqslant 1 \\ x^{2}, & x>1\end{cases}
$$

则 $f(x)$ 在 $x=1$ 处的 ( ).
(A) 左、右导数都存在.
(B) 左导数存在,右导数不存在.
(C) 左导数不存在,右导数存在
(D) 左、右导数都不存在.

8. 设 $f(x)$ 可导, $F(x)=f(x)(1+|\sin x|)$, 则 $f(0)=0$ 是 $F(x)$ 在 $x=0$ 处可导的 ( ).
(A) 充分必要条件.
（B）充分条件但非必要条件.
(C) 必要条件但非充分条件.
（D）既非充分条件又非必要条件.
9. 求下列函数的导数:
(1) $y=x^{4}$;
(2) $y=\sqrt[3]{x^{2}}$;
(3) $y=x^{1.6}$;
(4) $y=\frac{1}{\sqrt{x}}$;
(5) $y=\frac{1}{x^{2}}$;
(6) $y=x^{3} \sqrt[5]{x}$;

(7) $y=\frac{x^{2} \sqrt[3]{x^{2}}}{\sqrt{x^{5}}}$.

10. 已知物体的运动规律为 $s=t^{3}(\mathrm{~m})$, 求这物体在 $t=2(\mathrm{~s})$ 时的速度.
11. 如果 $f(x)$ 为偶函数, 且 $f^{\circ}(0)$ 存在, 证明 $f^{\prime}(0)=0$.
12. 求曲线 $y=\sin x$ 在具有下列横坐标的各点处切线的斜率:

$$
x=\frac{2}{3} \pi ; \quad x=\pi \text {. }
$$

13. 求曲线 $y=\cos x$ 上点 $\left(\frac{\pi}{3}, \frac{1}{2}\right)$ 处的切线方程和法线方程.
14. 求曲线 $y=\mathrm{e}^{x}$ 在点 $(0,1)$ 处的切线方程.
15. 在抛物线 $y=x^{2}$ 上取横坐标为 $x_{1}=1$ 及 $x_{2}=3$ 的两点, 作过这两点的割线. 问该拖 物线上哪一点的切线平行于这条割线?
16. 讨论下列函数在 $x=0$ 处的连续性与可导性:

(1) $y=|\sin x|$;

(2) $y= \begin{cases}x^{2} \sin \frac{1}{x}, & x \neq 0, \\ 0, & x=0 .\end{cases}$

17. 设函数

$$
f(x)= \begin{cases}x^{2}, & x \leqslant 1, \\ a x+b, & x>1 .\end{cases}
$$

为了使函数 $f(x)$ 在 $x=1$ 处连续且可导, $a, b$ 应取什么值?

18. 已知 $f(x)=\left\{\begin{array}{ll}x^{2}, & x \geqslant 0, \\ -x, & x<0,\end{array}\right.$ 求 $f_{+}(0)$ 及 $f^{\prime}-(0)$, 又 $f^{\prime}(0)$ 是否存在?
19. 已知 $f(x)=\left\{\begin{array}{ll}\sin x, & x<0, \\ x, & x \geqslant 0,\end{array}\right.$ 求 $f^{\prime}(x)$. 20. 证明: 双曲线 $x y=a^{2}$ 上任一点处的切线与两坐标轴构成的三角形的面积都等于 $2 a^{2}$.

## 第二节 函数的求导法则

在本节中, 将介绍求导数的几个基本法则以及前一节中末讨论过的几个基 本初等函数的导数公式. 借助于这些法则和基本初等函数的导数公式, 就能比较 方便地求出常见的初等函数的导数.

## 一、函数的和、差、积、商的求导法则

定理 1 如果函数 $u=u(x)$ 及 $v=v(x)$ 都在点 $x$ 具有导数,那么它们的 和、差、积、商 (除分母为零的点外) 都在点 $x$ 具有导数, 且

(1) $[u(x) \pm v(x)]^{\prime}=u^{\prime}(x) \pm v^{\prime}(x)$;

(2) $[u(x) v(x)]^{\prime}=u^{\prime}(x) v(x)+u(x) v^{\prime}(x)$;

(3) $\left[\frac{u(x)}{v(x)}\right]^{\prime}=\frac{u^{\prime}(x) v(x)-u(x) v^{\prime}(x)}{v^{2}(x)}(v(x) \neq 0)$.

证

$$
\text { (1) } \begin{aligned}
& {[u(x) \pm v(x)]^{\prime} } \\
= & \lim _{\Delta x \rightarrow 0} \frac{[u(x+\Delta x) \pm v(x+\Delta x)]-[u(x) \pm v(x)]}{\Delta x} \\
= & \lim _{\Delta x \rightarrow 0} \frac{u(x+\Delta x)-u(x)}{\Delta x} \pm \lim _{\Delta x \rightarrow 0} \frac{v(x+\Delta x)-v(x)}{\Delta x} \\
= & u^{\prime}(x) \pm v^{\prime}(x) .
\end{aligned}
$$

于是法则 (1) 获得证明. 法则 (1) 可简单地表示为

$$
(u \pm v)^{\prime}=u^{\prime} \pm v^{\prime} \text {. }
$$

(2) $[u(x) v(x)]^{\prime}$

$$
\begin{aligned}
& =\lim _{\Delta x \rightarrow 0} \frac{u(x+\Delta x) v(x+\Delta x)-u(x) v(x)}{\Delta x} \\
& =\lim _{\Delta x \rightarrow 0}\left[\frac{u(x+\Delta x)-u(x)}{\Delta x} \cdot v(x+\Delta x)+u(x) \cdot \frac{v(x+\Delta x)-v(x)}{\Delta x}\right] \\
& =\lim _{\Delta x \rightarrow 0} \frac{u(x+\Delta x)-u(x)}{\Delta x} \cdot \lim _{\Delta x \rightarrow 0} v(x+\Delta x)+u(x) \cdot \lim _{\Delta x \rightarrow 0} \frac{v(x+\Delta x)-v(x)}{\Delta x} \\
& =u^{\prime}(x) v(x)+u(x) v^{\prime}(x) .
\end{aligned}
$$

其中 $\lim _{\Delta x \rightarrow 0} v(x+\Delta x)=v(x)$ 是由于 $v^{\prime}(x)$ 存在, 故 $v(x)$ 在点 $x$ 连续. 于是法则 (2) 获得证明. 法则 (2) 可简单地表示为

$$
\begin{aligned}
& (u v)^{\prime}=u^{\prime} v+u v^{\prime} . \\
& \text { (3) }\left[\frac{u(x)}{v(x)}\right]^{\prime}=\lim _{\Delta x \rightarrow 0} \frac{\frac{u(x+\Delta x)}{v(x+\Delta x)}-\frac{u(x)}{v(x)}}{\Delta x} \\
& =\lim _{\Delta x \rightarrow 0} \frac{u(x+\Delta x) v(x)-u(x) v(x+\Delta x)}{v(x+\Delta x) v(x) \Delta x} \\
& =\lim _{\Delta x \rightarrow 0} \frac{[u(x+\Delta x)-u(x)] v(x)-u(x)[v(x+\Delta x)-v(x)]}{v(x+\Delta x) v(x) \Delta x} \\
& =\lim _{\Delta x \rightarrow 0} \frac{\frac{u(x+\Delta x)-u(x)}{\Delta x} v(x)-u(x) \frac{v(x+\Delta x)-v(x)}{\Delta x}}{v(x+\Delta x) v(x)} \\
& =\frac{u^{\prime}(x) v(x)-u(x) v^{\prime}(x)}{v^{2}(x)} .
\end{aligned}
$$

于是法则 (3) 获得证明. 法则 (3) 可简单地表示为

$$
\left(\frac{u}{v}\right)^{\prime}=\frac{u^{\prime} v-u v^{\prime}}{v^{2}} \text {. }
$$

定理 1 中的法则 (1)、(2) 可推广到任意有限个可导函数的情形. 例如, 设 $u$ $=u(x) 、 v=v(x) 、 w=w(x)$ 均可导, 则有

$$
\begin{aligned}
& (u+v-w)^{\prime}=u^{\prime}+v^{\prime}-w^{\prime}, \\
& (u v w)^{\prime}=[(u v) w]^{\prime}=(u v)^{\prime} w+(u v) w^{\prime}=\left(u^{\prime} v+u v^{\prime}\right) w+u v w^{\prime},
\end{aligned}
$$

即

$$
(u v w)^{\prime}=u^{\prime} v w+u v^{\prime} w+u v w w^{\prime} \text {. }
$$

在法则 (2) 中, 当 $v(x)=C$ ( $C$ 为常数) 时, 有

$$
(C u)^{\prime}=C u^{\prime} \text {. }
$$

例 $1 y=2 x^{3}-5 x^{2}+3 x-7$, 求 $y^{\prime}$.

解

$$
\begin{aligned}
y^{\prime} & =\left(2 x^{3}-5 x^{2}+3 x-7\right)^{\prime} \\
& =\left(2 x^{3}\right)^{\prime}-\left(5 x^{2}\right)^{\prime}+(3 x)^{\prime}-(7)^{\prime} \\
& =2 \cdot 3 x^{2}-5 \cdot 2 x+3-0=6 x^{2}-10 x+3 .
\end{aligned}
$$

例 $2 f(x)=x^{3}+4 \cos x-\sin \frac{\pi}{2}$, 求 $f^{\prime}(x)$ 及 $f^{\prime}\left(\frac{\pi}{2}\right)$.

解 $f^{\prime}(x)=3 x^{2}-4 \sin x$,

$$
f^{\prime}\left(\frac{\pi}{2}\right)=\frac{3}{4} \pi^{2}-4
$$

例 $3 y=\mathrm{e}^{x}(\sin x+\cos x)$, 求 $y^{\prime}$.

解 $y^{\prime}=\left(\mathrm{e}^{x}\right)^{\prime}(\sin x+\cos x)+\mathrm{e}^{x}(\sin x+\cos x)^{\prime}$

$$
\begin{aligned}
& =\mathrm{e}^{x}(\sin x+\cos x)+\mathrm{e}^{x}(\cos x-\sin x) \\
& =2 \mathrm{e}^{x} \cos x .
\end{aligned}
$$

例 $4 y=\tan x$, 求 $y^{\prime}$.

解 $y^{\prime}=(\tan x)^{\prime}=\left(\frac{\sin x}{\cos x}\right)^{\prime}$

$$
\begin{aligned}
& =\frac{(\sin x)^{\prime} \cos x-\sin x(\cos x)^{\prime}}{\cos ^{2} x} \\
& =\frac{\cos ^{2} x+\sin ^{2} x}{\cos ^{2} x}=\frac{1}{\cos ^{2} x}=\sec ^{2} x,
\end{aligned}
$$

即

$$
(\tan x)^{\prime}=\sec ^{2} x \text {. }
$$

这就是正切函数的导数公式.

例 $5 y=\sec x$, 求 $y^{\prime}$.

解 $y^{\prime}=(\sec x)^{\prime}=\left(\frac{1}{\cos x}\right)^{\prime}$

$$
\begin{aligned}
& =\frac{(1)^{\prime} \cos x-1 \cdot(\cos x)^{\prime}}{\cos ^{2} x} \\
& =\frac{\sin x}{\cos ^{2} x}=\sec x \tan x,
\end{aligned}
$$

即

$$
(\sec x)^{\prime}=\sec x \tan x .
$$

这就是正割函数的导数公式.

用类似方法, 还可求得余切函数及余割函数的导数公式

$$
\begin{aligned}
& (\cot x)^{\prime}=-\csc ^{2} x, \\
& (\csc x)^{\prime}=-\csc x \cot x .
\end{aligned}
$$

## 二、反函数的求导法则

定理 2 如果函数 $x=f(y)$ 在区间 $I_{y}$ 内单调、可导且 $f^{\prime}(y) \neq 0$, 则它的反 函数 $y=f^{-1}(x)$ 在区间 $I_{x}=\left\{x \mid x=f(y), y \in I_{y}\right\}$ 内也可导, 且

$$
\left[f^{-1}(x)\right]^{\prime}=\frac{1}{f^{\prime}(y)} \text { 或 } \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{\frac{\mathrm{d} x}{\mathrm{~d} y}} \text {. }
$$

证 由于 $x=f(y)$ 在 $I_{y}$ 内单调、可导 (从而连续), 由第一章第九节定理 2 知道, $x=f(y)$ 的反函数 $y=f^{-1}(x)$ 存在, 且 $f^{-1}(x)$ 在 $I_{x}$ 内也单调、连续.

任取 $x \in I_{x}$, 给 $x$ 以增量 $\Delta x\left(\Delta x \neq 0, x+\Delta x \in I_{x}\right)$, 由 $y=f^{-1}(x)$ 的单调 性可知

$$
\Delta y=f^{-1}(x+\Delta x)-f^{-1}(x) \neq 0,
$$

于是有

$$
\frac{\Delta y}{\Delta x}=\frac{1}{\frac{\Delta x}{\Delta y}}
$$

因 $y=f^{-1}(x)$ 连续, 故

$$
\lim _{\Delta x \rightarrow 0} \Delta y=0
$$

从而

$$
\left[f^{-1}(x)\right]^{\prime}=\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=\lim _{\Delta y \rightarrow 0} \frac{1}{\frac{\Delta x}{\Delta y}}=\frac{1}{f^{\prime}(y)} .
$$

上述结论可简单地说成: 反函数的导数等于直接函数导数的倒数.

下面用上述结论来求反三角函数及对数函数的导数.

例 6 设 $x=\sin y, y \in\left[-\frac{\pi}{2}, \frac{\pi}{2}\right]$ 为直接函数, 则 $y=\arcsin x$ 是它的反函 数. 函数 $x=\sin y$ 在开区间 $I_{y}=\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ 内单调、可导, 且

$$
(\sin y)^{\prime}=\cos y>0 \text {. }
$$

因此,由公式(1), 在对应区间 $I_{x}=(-1,1)$ 内有

$$
(\arcsin x)^{\prime}=\frac{1}{(\sin y)^{\prime}}=\frac{1}{\cos y} .
$$

但 $\cos y=\sqrt{1-\sin ^{2} y}=\sqrt{1-x^{2}}$ (因为当 $-\frac{\pi}{2}<y<\frac{\pi}{2}$ 时, $\cos y>0$, 所以根号前 只取正号), 从而得反正弦函数的导数公式

$$
(\arcsin x)^{\prime}=\frac{1}{\sqrt{1-x^{2}}} .
$$

用类似的方法可得反余弦函数的导数公式

$$
(\arccos x)^{\prime}=-\frac{1}{\sqrt{1-x^{2}}} .
$$

例 7 设 $x=\tan y$ 是直接函数, $y \in I_{y}=\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$, 则 $y=\arctan x$ 是它的 反函数. 函数 $x=\tan y$ 在 $I_{y}=\left(-\frac{\pi}{2}, \frac{\pi}{2}\right)$ 内单调、可导, 且

$$
(\tan y)^{\prime}=\sec ^{2} y \neq 0 \text {. }
$$

因此, 由公式 (1), 在对应区间 $I_{x}=(-\infty,+\infty)$ 内有

$$
(\arctan x)^{\prime}=\frac{1}{(\tan y)^{\prime}}=\frac{1}{\sec ^{2} y} \text {. }
$$

但 $\sec ^{2} y=1+\tan ^{2} y=1+x^{2}$, 从而得反正切函数的导数公式

$$
(\arctan x)^{\prime}=\frac{1}{1+x^{2}} .
$$

用类似的方法可得反余切函数的导数公式

$$
(\operatorname{arccot} x)^{\prime}=-\frac{1}{1+x^{2}} \text {. }
$$

如果利用三角学中的公式

$\arccos x=\frac{\pi}{2}-\arcsin x$ 和 $\operatorname{arccot} x=\frac{\pi}{2}-\arctan x$, 那么从本节公式(2)和 (4),也立刻可得公式(3)和(5).

例 8 设 $x=a^{y}(a>0, a \neq 1)$ 为直接函数, 则 $y=\log _{a} x$ 是它的反函数. 函数 $x=a^{y}$ 在区间 $I_{y}=(-\infty,+\infty)$ 内单调、可导, 且

$$
\left(a^{v}\right)^{\prime}=a^{y} \ln a \neq 0 \text {. }
$$

因此,由公式 (1), 在对应区间 $I_{s}=(0,+\infty)$ 内有

$$
\left(\log _{a} x\right)^{\prime}=\frac{1}{\left(a^{y}\right)^{\prime}}=\frac{1}{a^{y} \ln a} \text {. }
$$

但 $a^{y}=x$, 从而得到第一节例 5 中已求得的对数函数的导数公式

$$
\left(\log _{a} x\right)^{\prime}=\frac{1}{x \ln a} \text {. }
$$

## 三、复合函数的求导法则

到目前为止, 对于

$$
\ln \tan x, \quad \mathrm{e}^{x^{3}}, \quad \sin \frac{2 x}{1+x^{2}}
$$

那样的函数,我们还不知道它们是否可导, 可导的话如何求它们的导数.这些问 题借助于下面的重要法则可以得到解决, 从而使可以求得导数的函数的范围得 到很大扩充.

定理 3 如果 $u=g(x)$ 在点 $x$ 可导, 而 $y=f(u)$ 在点 $u=g(x)$ 可导, 则复 合函数 $y=f[g(x)]$ 在点 $x$ 可导, 且其导数为

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=f^{\prime}(u) \cdot g^{\prime}(x) \text { 或 } \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} u} \cdot \frac{\mathrm{d} u}{\mathrm{~d} x} .
$$

证 由于 $y=f(u)$ 在点 $u$ 可导, 因此

$$
\lim _{\Delta u \rightarrow 0} \frac{\Delta y}{\Delta u}=f^{\prime}(u)
$$

存在, 于是根据极限与无穷小的关系有

$$
\frac{\Delta y}{\Delta u}=f^{\prime}(u)+\alpha,
$$

其中 $\alpha$ 是 $\Delta u \rightarrow 0$ 时的无穷小. 上式中 $\Delta u \neq 0$,用 $\Delta u$ 乘上式两边,得

$$
\Delta y=f^{\prime}(u) \Delta u+\alpha \cdot \Delta u \text {. }
$$

当 $\Delta u=0$ 时, 规定 $a=0^{\mathbb{D}}$, 这时因 $\Delta y=f(u+\Delta u)-f(u)=0$, 而 (7) 式右端亦 为零, 故 (7) 式对 $\Delta u=0$ 也成立. 用 $\Delta x \neq 0$ 除(7)式两边, 得

$$
\frac{\Delta y}{\Delta x}=f^{\prime}(u) \frac{\Delta u}{\Delta x}+\alpha \cdot \frac{\Delta u}{\Delta x},
$$

于是

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=\lim _{\Delta x_{1} \rightarrow 0}\left[f^{\prime}(u) \frac{\Delta u}{\Delta x}+\alpha \frac{\Delta u}{\Delta x}\right] .
$$

根据函数在某点可导必在该点连续的性质知道, 当 $\Delta x \rightarrow 0$ 时, $\Delta u \rightarrow 0$, 从而可以 推知

又因 $u=g(x)$ 在点 $x$ 处可导, 有

$$
\lim _{\Delta, r \rightarrow 0} \alpha=\lim _{\Delta u \rightarrow 0} \alpha=0 .
$$

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta u}{\Delta x}=g^{\prime}(x),
$$

故

$$
\begin{gathered}
\lim _{\Delta x \rightarrow 0 \rightarrow} \frac{\Delta y}{\Delta x}=f^{\prime}(u) \cdot \lim _{\Delta x \rightarrow i 1} \frac{\Delta u}{\Delta x}, \\
\frac{\mathrm{d} y}{\mathrm{~d} x}=f^{\prime}(u) \cdot g^{\prime}(x) .
\end{gathered}
$$

即

这就是公式(6).

例 $9 y=\mathrm{e}^{x^{3}}$, 求 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 $y=\mathrm{e}^{x^{3}}$ 可看作由 $y=\mathrm{e}^{u}, u=x^{3}$ 复合而成, 因此

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} u} \cdot \frac{\mathrm{d} u}{\mathrm{~d} x}=\mathrm{e}^{u} \cdot 3 x^{2}=3 x^{2} \mathrm{e}^{x^{3}} .
$$

例 $10 y=\sin \frac{2 x}{1+x^{2}}$, 求 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 $y=\sin \frac{2 x}{1+x^{2}}$ 可看作由 $y=\sin u, u=\frac{2 x}{1+x^{2}}$ 复合而成. 因

$$
\begin{aligned}
& \frac{\mathrm{d} y}{\mathrm{~d} u}=\cos u, \\
& \frac{\mathrm{d} u}{\mathrm{~d} x}=\frac{2\left(1+x^{2}\right)-(2 x)^{2}}{\left(1+x^{2}\right)^{2}}=\frac{2\left(1-x^{2}\right)}{\left(1+x^{2}\right)^{2}},
\end{aligned}
$$

所以

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\cos u \cdot \frac{2\left(1-x^{2}\right)}{\left(1+x^{2}\right)^{2}}=\frac{2\left(1-x^{2}\right)}{\left(1+x^{2}\right)^{2}} \cdot \cos \frac{2 x}{1+x^{2}} .
$$

从以上例子看出, 应用复合函数求导法则时,首先要分析所给函数可看作由

(1) $\alpha=\frac{\Delta y}{\Delta u}-f(u)$. 印 $a=a(\Delta u)$. 这雨数当 $\Delta u=0$ 时无定义, 当 $\Delta u \rightarrow 0$ 时, $\alpha \rightarrow 0$. 今规定 $\Delta u=0$ 时 $a=0$, 则该函数在 $\Delta u=0$ 处连绕. 哪些函数复合而成,或者说,所给函数能分解成哪些函数.如果所给函数能分解 成比较简单的函数,而这些简单函数的导数我们已经会求,那么应用复合函数求 导法则就可以求所给函数的导数了.

对复合函数的分解比较熟练后,就不必再写出中间变量, 而可以采用下列例 题的方式来计算.

例 $11 y=\ln \sin x$, 求 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 $\frac{\mathrm{d} y}{\mathrm{~d} x}=(\ln \sin x)^{\prime}=\frac{1}{\sin x}(\sin x)^{\prime}=\frac{\cos x}{\sin x}=\cot x$.

例 $12 y=\sqrt[3]{1-2 x^{2}}$, 求 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 $\frac{\mathrm{d} y}{\mathrm{~d} x}=\left[\left(1-2 x^{2}\right)^{\frac{1}{3}}\right]^{\prime}=\frac{1}{3}\left(1-2 x^{2}\right)^{-\frac{2}{3}} \cdot\left(1-2 x^{2}\right)^{\prime}$

$$
=\frac{-4 x}{3 \sqrt[3]{\left(1-2 x^{2}\right)^{2}}} \text {. }
$$

复合函数的求导法则可以推广到多个中间变量的情形.我们以两个中间变 量为例，设 $y=f(u), u=\varphi(v), v=\psi(x)$, 则

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} u} \cdot \frac{\mathrm{d} u}{\mathrm{~d} x},
$$

而 $\frac{\mathrm{d} u}{\mathrm{~d} x}=\frac{\mathrm{d} u}{\mathrm{~d} v} \cdot \frac{\mathrm{d} v}{\mathrm{~d} x}$, 故复合函数 $y=f\{\varphi[\psi(x)]\}$ 的导数为

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} u} \cdot \frac{\mathrm{d} u}{\mathrm{~d} v} \cdot \frac{\mathrm{d} v}{\mathrm{~d} x} .
$$

当然,这里假定上式右端所出现的导数在相应处都存在.

例 $13 y=\ln \cos \left(\mathrm{e}^{x}\right)$, 求 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 所给函数可分解为 $y=\ln u, u=\cos v, v=\mathrm{e}^{x}$. 因 $\frac{\mathrm{d} y}{\mathrm{~d} u}=\frac{1}{u}, \frac{\mathrm{d} u}{\mathrm{~d} v}=$ $-\sin v, \frac{\mathrm{d} v}{\mathrm{~d} x}=\mathrm{e}^{x}$, 故

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{u} \cdot(-\sin v) \cdot \mathrm{e}^{x}=-\frac{\sin \left(\mathrm{e}^{x}\right)}{\cos \left(\mathrm{e}^{x}\right)} \cdot \mathrm{e}^{x}=-\mathrm{e}^{x} \tan \left(\mathrm{e}^{x}\right) .
$$

不写出中间变量,此例可这样写:

$$
\begin{aligned}
\frac{\mathrm{d} y}{\mathrm{~d} x} & =\left[\ln \cos \left(\mathrm{e}^{x}\right)\right]^{\prime}=\frac{1}{\cos \left(\mathrm{e}^{x}\right)}\left[\cos \left(\mathrm{e}^{x}\right)\right]^{\prime} \\
& =\frac{-\sin \left(\mathrm{e}^{x}\right)}{\cos \left(\mathrm{e}^{x}\right)}\left(\mathrm{e}^{x}\right)^{\prime}=-\mathrm{e}^{-x} \tan \left(\mathrm{e}^{x}\right) .
\end{aligned}
$$

例 $14 y=\mathrm{e}^{\mathrm{N} N n^{\prime}} \cdot \frac{1}{r}$, 求 $y^{\prime}$. 解 $y^{\prime}=\left(\mathrm{e}^{\sin 1 \frac{1}{x}}\right)^{\prime}=\mathrm{e}^{\sin \frac{1}{x}}\left(\sin \frac{1}{x}\right)^{\prime}$

$$
=\mathrm{e}^{\sin \frac{1}{x}} \cdot \cos \frac{1}{x} \cdot\left(\frac{1}{x}\right)^{\prime}=-\frac{1}{x^{2}} \mathrm{e}^{\sin \frac{1}{x}} \cdot \cos \frac{1}{x} .
$$

例 15 设 $x>0$,证明幂函数的导数公式

$$
\left(x^{\prime \prime}\right)^{\prime}=\mu x^{\prime \prime-1} .
$$

证 因为 $x^{\mu}=\left(\mathrm{e}^{\ln x}\right)^{\mu}=\mathrm{e}^{\mu \ln x}$, 所以

$$
\begin{aligned}
\left(x^{\mu}\right)^{\prime} & =\left(\mathrm{e}^{\mu \ln x}\right)^{\prime}=\mathrm{e}^{\mu \ln x} \cdot(\mu \ln x)^{\prime} \\
& =x^{\mu} \cdot \mu \cdot \frac{1}{x}=\mu x^{\mu-1} .
\end{aligned}
$$

## 四、基本求导法则与导数公式

基本初等函数的导数公式与本节中所讨论的求导法则, 在初等函数的求导 运算中起着重要的作用,我们必须熟练地掌握它们. 为了便于查阅, 现在把这些 导数公式和求导法则归纳如下:

## 1. 常数和基本初等函数的导数公式

(1) $(C)^{\prime}=0$,
(2) $\left(x^{\prime \prime}\right)^{\prime}=\mu x^{\prime \prime-1}$,
(3) $(\sin x)^{\prime}=\cos x$,
(4) $(\cos x)^{\prime}=-\sin x$,
(5) $(\tan x)^{\prime}=\sec ^{2} x$,
(6) $(\cot x)^{\prime}=-\csc ^{2} x$,
(7) $(\sec x)^{\prime}=\sec x \tan x$,
(8) $(\csc x)^{\prime}=-\csc x \cot x$,
(9) $\left(a^{x}\right)^{\prime}=a^{x} \ln a$,
(10) $\left(\mathrm{e}^{x}\right)^{\prime}=\mathrm{e}^{x}$,
(11) $\left(\log _{a} x\right)^{\prime}=\frac{1}{x \ln a}$,
(12) $(\ln x)^{\prime}=\frac{1}{x}$,
(13) $(\arcsin x)^{\prime}=\frac{1}{\sqrt{1-x^{2}}}$,
(14) $(\arccos x)^{\prime}=-\frac{1}{\sqrt{1-x^{2}}}$,
(15) $(\arctan x)^{\prime}=\frac{1}{1+x^{2}}$,
(16) $(\operatorname{arccot} x)^{\prime}=-\frac{1}{1+x^{2}}$.

2. 函数的和、差、积、商的求导法则

设 $u=u(x), v=v(x)$ 都可导, 则
(1) $(u \pm v)^{\prime}=u^{\prime} \pm v^{\prime}$,
(2) $(C u)^{\prime}=C u^{\prime}$ ( $C$ 是常数),
(3) $(u v)^{\prime}=u^{\prime} v+u v^{\prime}$,
(4) $\left(\frac{u}{v}\right)^{\prime}=\frac{u^{\prime} v-u v^{\prime}}{v^{2}}(v \neq 0)$.

## 3. 反函数的求导法则

设 $x=f(y)$ 在区间 $I_{y}$ 内单调、可导且 $f^{\prime}(y) \neq 0$, 则它的反函数 $y=f^{-1}(x)$ 在 $I_{x}=f\left(I_{x}\right)$ 内也可导, 且

$$
\left[f^{-1}(x)\right]^{\prime}=\frac{1}{f^{\prime}(y)} \text { 或 } \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{\frac{\mathrm{d} x}{\mathrm{~d} y}} .
$$

## 4. 复合函数的求导法则

设 $y=f(u)$, 而 $u=g(x)$ 且 $f(u)$ 及 $g(x)$ 都可导, 则复合函数 $y=$ $f[g(x)]$ 的导数为

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} u} \cdot \frac{\mathrm{d} u}{\mathrm{~d} x} \text { 或 } y^{\prime}(x)=f^{\prime}(u) \cdot g^{\prime}(x) .
$$

下面再举两个综合运用这些法则和导数公式的例子.

例 $16 y=\sin n x \cdot \sin ^{n} x$ ( $n$ 为常数), 求 $y^{\prime}$.

解 首先应用积的求导法则得

$$
y^{\prime}=(\sin n x)^{\prime} \sin ^{\prime \prime} x+\sin n x \cdot\left(\sin ^{\prime \prime} x\right)^{\prime} .
$$

在计算 $(\sin n x)^{\prime}$ 与 $\left(\sin ^{\prime \prime} x\right)^{\prime}$ 时, 都要应用复合函数求导法则, 由此得

$$
\begin{aligned}
y^{\prime} & =n \cos n x \cdot \sin ^{n} x+\sin n x \cdot n \sin ^{n-1} x \cdot \cos x \\
& =n \sin ^{n-1} x(\cos n x \cdot \sin x+\sin n x \cdot \cos x) \\
& =n \sin ^{n-1} x \cdot \sin (n+1) x .
\end{aligned}
$$

"例 17 证明下列双曲函数及反双曲函数的导数公式

$$
\begin{aligned}
& (\operatorname{sh} x)^{\prime}=\operatorname{ch} x,(\operatorname{ch} x)^{\prime}=\operatorname{sh} x,(\operatorname{th} x)^{\prime}=\frac{1}{\operatorname{ch}^{2} x}, \\
& (\operatorname{arch} x)^{\prime}=\frac{1}{\sqrt{1+x^{2}}},(\operatorname{arch} x)^{\prime}=\frac{1}{\sqrt{x^{2}-1}},(\operatorname{arth} x)^{\prime}=\frac{1}{1-x^{2}} .
\end{aligned}
$$

证 由定理 1(1)、(2), 有

$$
(\operatorname{sh} x)^{\prime}=\left(\frac{\mathrm{e}^{x}-\mathrm{e}^{-x}}{2}\right)^{\prime}=\frac{\left(\mathrm{e}^{x}\right)^{\prime}-\left(\mathrm{e}^{-x}\right)^{\prime}}{2},
$$

再利用 $\left(\mathrm{e}^{x}\right)^{\prime}=\mathrm{e}^{x}$ 及定理 3 , 得 $\left(\mathrm{e}^{-x}\right)^{\prime}=-\mathrm{e}^{-x}$. 于是

$$
(\operatorname{sh} x)^{\prime}=\frac{\left(\mathrm{e}^{x}\right)^{\prime}-\left(\mathrm{e}^{-x}\right)^{\prime}}{2}=\frac{\mathrm{e}^{x}+\mathrm{e}^{-. r}}{2}=\operatorname{ch} x .
$$

同理可得

$$
(\operatorname{ch} x)^{\prime}=\left(\frac{\mathrm{e}^{x}+\mathrm{e}^{-x}}{2}\right)^{\prime}=\frac{\mathrm{e}^{x}-\mathrm{e}^{-x}}{2}=\operatorname{sh} x .
$$

由定理 1(3)及上述结果, 有

$$
(\operatorname{th} x)^{\prime}=\left(\frac{\operatorname{sh} x}{\operatorname{ch} x}\right)^{\prime}=\frac{(\operatorname{sh} x)^{\prime} \operatorname{ch} x-\operatorname{sh} x(\operatorname{ch} x)^{\prime}}{\operatorname{ch}^{2} x}
$$

$$
=\frac{\operatorname{ch}^{2} x-\operatorname{sh}^{2} x}{\operatorname{ch}^{2} x}=\frac{1}{\operatorname{ch}^{2} x} .
$$

由 $\operatorname{arsh} x=\ln \left(x+\sqrt{1+x^{2}}\right)$, 应用复合函数求导法则及定理 1(1), 有

$$
\begin{aligned}
(\operatorname{arsh} x)^{\prime} & =\frac{1}{x+\sqrt{1+x^{2}}}\left(x+\sqrt{1+x^{2}}\right)^{\prime} \\
& =\frac{1}{x+\sqrt{1+x^{2}}}\left(1+\frac{x}{\sqrt{1+x^{2}}}\right)=\frac{1}{\sqrt{1+x^{2}}} .
\end{aligned}
$$

由 $\operatorname{arch} x=\ln \left(x+\sqrt{x^{2}-1}\right)$, 同理可得

$$
(\operatorname{arch} x)^{\prime}=\frac{1}{\sqrt{x^{2}-1}}, x \in(1,+\infty) \text {. }
$$

由 $\operatorname{arth} x=\frac{1}{2} \ln \frac{1+x}{1-x}$, 可得

$$
(\operatorname{arth} x)^{\prime}=\frac{1}{1-x^{2}}, x \in(-1,1) \text {. }
$$

## 习 题 2-2

## 1. 推导余切函数及余割函数的导数公式:

$$
(\cot x)^{\prime}=-\csc ^{2} x, \quad(\csc x)^{\prime}=-\csc x \cot x .
$$

2. 求下列函数的导数:
(1) $y=x^{3}+\frac{7}{x^{4}}-\frac{2}{x}+12$;
(2) $y=5 x^{3}-2 x+3 e^{y}$;
(3) $y=2 \tan x+\sec x-1$;
(4) $y=\sin x \cdot \cos x$;
(5) $y=x^{2} \ln x$;
(6) $y=3 \mathrm{e}^{r} \cos x$;
(7) $y=\frac{\ln x}{x}$;
(8) $y=\frac{\mathrm{e}^{x}}{x^{2}}+\ln 3$;
(9) $y=x^{2} \ln x \cos x$;
(10) $s=\frac{1+\sin t}{1+\cos t}$.
3. 求下列函数在给定点处的导数:
(1) $y=\sin x-\cos x$, 求 $\left.y^{\prime}\right|_{x-\frac{x}{6}}$ 和 $\left.y^{\prime}\right|_{x=\frac{x}{T}}$;
(2) $\rho=\theta \sin \theta+\frac{1}{2} \cos \theta$, 求 $\left.\frac{\mathrm{d} \rho}{\mathrm{d} \theta}\right|_{\theta=\frac{\pi}{T}}$;
(3) $f(x)=\frac{3}{5-x}+\frac{x^{2}}{5}$, 求 $f(0)$ 和 $f(2)$.
4. 以初速度 $v_{11}$ 坚直上地的物体, 其上升缡度 $s$ 与时问 $t$ 的关系是 $s=v_{0} t-\frac{1}{2} g t^{2}$. 求:
(1) 该物体的速度 $v(t)$;
（2）该物体达到很高点的时刻. 5. 求曲线 $y=2 \sin x+x^{2}$ 上椛坐标为 $x=0$ 的点处的切线方程和法线方程.
5. 求下列函数的导数:
(1) $y=(2 x+5)^{4}$;
(2) $y=\cos (4-3 x)$;
(3) $y=\mathrm{e}^{-3 . r^{2}}$;
(4) $y=\ln \left(1+x^{2}\right)$;
(5) $y=\sin ^{2} x$;
(6) $y=\sqrt{a^{2}-x^{2}}$;
(7) $y=\tan x^{2}$;
(8) $y=\arctan \left(\mathrm{e}^{\prime}\right)$;
(9) $y=(\arcsin x)^{2}$;
(10) $y=\ln \cos x$.
6. 求下列函数的导数:
(1) $y=\arcsin (1-2 x)$;
(2) $y=\frac{1}{\sqrt{1-x^{2}}}$;
(3) $y=\mathrm{e}^{-\frac{1}{2}} \cos 3 x$;
(4) $y=\arccos \frac{1}{x}$;
(5) $y=\frac{1-\ln x}{1+\ln x}$;
(6) $y=\frac{\sin 2 x}{x}$;
(7) $y=\arcsin \sqrt{x}$;
(8) $y=\ln \left(x+\sqrt{a^{2}+x^{2}}\right)$;
(9) $y=\ln (\sec x+\tan x)$;
(10) $y=\ln (\csc x-\cot x)$.
7. 求下列函数的导数:
(1) $y=\left(\arcsin \frac{x}{2}\right)^{2}$;
(2) $y=\ln \tan \frac{x}{2}$;
(3) $y=\sqrt{1+\ln ^{2} x}$;
(4) $y=\mathrm{e}^{\operatorname{nuctrun} \sqrt{\mathrm{r}}}$;
(5) $y=\sin ^{n} x \cos n x$;
(6) $y=\arctan \frac{x+1}{x-1}$;
(7) $y=\frac{\arcsin x}{\arccos x}$
(8) $y=\ln \ln \ln x$;
(9) $y=\frac{\sqrt{1+x}-\sqrt{1-x}}{\sqrt{1+x}+\sqrt{1-x}}$;
(10) $y=\arcsin \sqrt{\frac{1-x}{1+x}}$.
8. 设函数 $f(x)$ 和 $g(x)$ 可导, 且 $f^{2}(x)+g^{2}(x) \neq 0$, 试求函数 $y=\sqrt{f^{2}(x)+g^{2}(x)}$ 的 导数.
9. 设 $f(x)$ 可导,求下列函数的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ :
(1) $y=f\left(x^{2}\right)$;
(2) $y=f\left(\sin ^{2} x\right)+f\left(\cos ^{2} x\right)$.
10. 求下列函数的导数:
(1) $y=\mathrm{e}^{-x}\left(x^{2}-2 x+3\right)$;
(2) $y=\sin ^{2} x \cdot \sin \left(x^{2}\right)$;
(3) $y=\left(\arctan \frac{x}{2}\right)^{2}$;
(4) $y=\frac{\ln x}{x^{\mu}}$;
(5) $y=\frac{\mathrm{e}^{\prime}-\mathrm{e}^{-1}}{\mathrm{e}^{\prime}+\mathrm{e}^{-1}}$;
(6) $y=\ln \cos \frac{1}{x}$;
(7) $y=\mathrm{e}^{-\sin ^{2} \frac{1}{1}}$;
(8) $y=\sqrt{x+\sqrt{x}}$;
(9) $y=x \arcsin \frac{x}{2}+\sqrt{4-x^{2}}$;
(10) $y=\arcsin \frac{2 t}{1+t^{2}}$. 12. 求下列函数的导数:
(1) $y=\operatorname{ch}(\operatorname{sh} x)$;
(2) $y=\operatorname{sh} x \cdot \mathrm{e}^{\mathrm{ch} x}$;
(3) $y=\operatorname{th}(\ln x)$;
(4) $y=\operatorname{sh}^{3} x+\operatorname{ch}^{2} x$;
(5) $y=\operatorname{th}\left(1-x^{2}\right)$;
(6) $y=\operatorname{arsh}\left(x^{2}+1\right)$;
(7) $y=\operatorname{arch}\left(\mathrm{e}^{2 . r}\right)$;
(8) $y=\arctan (\operatorname{th} x)$;
(9) $y=\ln \operatorname{ch} x+\frac{1}{2 \operatorname{ch}^{2} x}$;
(10) $y=\operatorname{ch}^{2}\left(\frac{x-1}{x+1}\right)$.
11. 设函数 $f(x)$ 和 $g(x)$ 均在点 $x_{0}$ 的某一邻域内有定义, $f(x)$ 在 $x_{11}$ 处可导, $f\left(x_{11}\right)=$ $0, g(x)$ 在 $x_{\mathrm{a}}$ 处连续, 试讨论 $f(x) g(x)$ 在 $x_{\mathrm{u}}$ 处的可导性.
12. 设函数 $f(x)$ 满足下列条件:

(1) $f(x+y)=f(x) \cdot f(y)$, 对一切 $x, y \in \mathbf{R}$;

(2) $f(x)=1+x g(x)$, 而 $\lim _{x \rightarrow 0} g(x)=1$.

试证明 $f(x)$ 在 R 上处处可导, 且 $f^{\prime}(x)=f(x)$.

## 第三节 高阶 导 数

我们知道, 变速直线运动的速度 $v(t)$ 是位置函数 $s(t)$ 对时间 $t$ 的导数, 即

$$
v=\frac{\mathrm{d} s}{\mathrm{~d} t} \text { 或 } v=s^{\prime} ，
$$

而加速度 $a$ 又是速度 $v$ 对时间 $t$ 的变化率, 即速度 $v$ 对时间 $t$ 的导数：

$$
a=\frac{\mathrm{d} v}{\mathrm{~d} t}=\frac{\mathrm{d}}{\mathrm{d} t}\left(\frac{\mathrm{d} s}{\mathrm{~d} t}\right) \text { 或 } a=\left(s^{\prime}\right)^{\prime} .
$$

这种导数的导数 $\frac{\mathrm{d}}{\mathrm{d} t}\left(\frac{\mathrm{d} s}{\mathrm{~d} t}\right)$ 或 $\left(s^{\prime}\right)^{\prime}$ 叫做 $s$ 对 $t$ 的云阶导数, 记作

$$
\frac{\mathrm{d}^{2} s}{\mathrm{~d} t^{2}} \text { 或 } s^{\prime \prime}(t) \text {. }
$$

所以, 直线运动的加速度就是位置函数 $s$ 对时间 $t$ 的二阶导数.

一般的, 函数 $y=f(x)$ 的导数 $y^{\prime}=f^{\prime}(x)$ 仍然是 $x$ 的函数. 我们把 $y^{\prime}=$ $f^{\prime}(x)$ 的导数叫做函数 $y=f(x)$ 的三阶导数, 记作 $y^{\prime \prime}$ 或 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$, 即

$$
y^{\prime \prime}=\left(y^{\prime}\right)^{\prime} \text { 或 } \frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}=\frac{\mathrm{d}}{\mathrm{d} x}\left(\frac{\mathrm{d} y}{\mathrm{~d} x}\right) .
$$

相应的, 把 $y=f^{\prime}(x)$ 的导数 $f^{\prime}(x)$ 叫做函数 $y=f(x)$ 的一阶导数.

类似地,二阶导数的导数, 叫做三阶导数,三阶导数的导数叫做四阶 导数, $\cdots$,一般的, $(n-1)$ 阶导数的导数叫做 $n$ 阶导数, 分别记作 或

$$
\begin{gathered}
y^{\prime \prime \prime}, y^{(4)}, \cdots, y^{(n)} \\
\frac{\mathrm{d}^{3} y}{\mathrm{~d} x^{3}}, \frac{\mathrm{d}^{4} y}{\mathrm{~d} x^{4}}, \cdots, \frac{\mathrm{d}^{\prime \prime} y}{\mathrm{~d} x^{\prime \prime}} .
\end{gathered}
$$

函数 $y=f(x)$ 具有 $n$ 阶导数, 也常说成函数 $f(x)$ 为 $n$ 阶可导. 如果函数 $f(x)$ 在点 $x$ 处具有 $n$ 阶导数, 那么 $f(x)$ 在点 $x$ 的某一邻域内必定具有一切低 于 $n$ 阶的导数.二阶及二阶以上的导数统称高阶导数.

由此可见,求高阶导数就是多次接连地求导数. 所以,仍可应用前面学过的 求导方法来计算高阶导数.

例 $1 y=a x+b$, 求 $y^{\prime \prime}$.

解 $y^{\prime}=a, y^{\prime \prime}=0$.

例 $2 s=\sin \omega t$, 求 $s$.

解 $s^{\prime}=\omega \cos \omega t, s^{\prime \prime}=-\omega^{2} \sin \omega t$.

例 3 证明: 函数 $y=\sqrt{2 x-x^{2}}$ 满足关系式

$$
y^{3} y^{\prime \prime}+1=0 \text {. }
$$

证 将 $y=\sqrt{2 x-x^{2}}$ 求导, 得

$$
\begin{aligned}
y^{\prime} & =\frac{2-2 x}{2 \sqrt{2 x-x^{2}}}=\frac{1-x}{\sqrt{2 x-x^{2}}}, \\
y^{\prime \prime} & =\frac{-\sqrt{2 x-x^{2}}-(1-x) \frac{2-2 x}{2 \sqrt{2 x-x^{2}}}}{2 x-x^{2}} \\
& =\frac{-2 x+x^{2}-(1-x)^{2}}{\left(2 x-x^{2}\right) \sqrt{2 x-x^{2}}} \\
& =-\frac{1}{\left(2 x-x^{2}\right)^{\frac{3}{2}}}=-\frac{1}{y^{3}} .
\end{aligned}
$$

于是

$$
y^{3} y^{\prime \prime}+1=0 \text {. }
$$

下面介绍几个初等函数的 $n$ 阶导数.

例 4 求指数函数 $y=\mathrm{e}^{x}$ 的 $n$ 阶导数.

解 $y^{\prime}=\mathrm{e}^{x}, y^{\prime \prime}=\mathrm{e}^{x}, y^{\prime \prime \prime}=\mathrm{e}^{x}, y^{(4)}=\mathrm{e}^{x}$.

一般的,可得

即

$$
\begin{gathered}
y^{(n)}=\mathrm{e}^{x}, \\
\left(\mathrm{e}^{x}\right)^{(n)}=\mathrm{e}^{x} .
\end{gathered}
$$

例 5 求正弦函数与余弦函数的 $n$ 阶导数.

解 $y=\sin x$,

$$
y^{\prime}=\cos x=\sin \left(x+\frac{\pi}{2}\right) \text {, }
$$

$$
\begin{aligned}
& y^{\prime \prime}=\cos \left(x+\frac{\pi}{2}\right)=\sin \left(x+\frac{\pi}{2}+\frac{\pi}{2}\right)=\sin \left(x+2 \cdot \frac{\pi}{2}\right), \\
& y^{\prime \prime}=\cos \left(x+2 \cdot \frac{\pi}{2}\right)=\sin \left(x+3 \cdot \frac{\pi}{2}\right), \\
& y^{(4)}=\cos \left(x+3 \cdot \frac{\pi}{2}\right)=\sin \left(x+4 \cdot \frac{\pi}{2}\right),
\end{aligned}
$$

一般的,可得

$$
y^{(n)}=\sin \left(x+n \cdot \frac{\pi}{2}\right),
$$

即

$$
(\sin x)^{(n)}=\sin \left(x+n \cdot \frac{\pi}{2}\right) \text {. }
$$

用类似方法, 可得

$$
(\cos x)^{(n)}=\cos \left(x+n \cdot \frac{\pi}{2}\right) .
$$

例 6 求函数 $\ln (1+x)$ 的 $n$ 阶导数.

解

$$
\begin{aligned}
& \text { 解 } y=\ln (1+x), y^{\prime}=\frac{1}{1+x}, \\
& y^{\prime \prime}=-\frac{1}{(1+x)^{2}}, \quad y^{\prime \prime \prime}=\frac{1 \cdot 2}{(1+x)^{3}}, \quad y^{(4)}=-\frac{1 \cdot 2 \cdot 3}{(1+x)^{4}},
\end{aligned}
$$

一般的, 可得

$$
y^{(n)}=(-1)^{n-1} \frac{(n-1) !}{(1+x)^{n}}
$$

即

$$
[\ln (1+x)]^{(n)}=(-1)^{n-1} \frac{(n-1) !}{(1+x)^{n}} .
$$

通常规定 $0 !=1$, 所以这个公式当 $n=1$ 时也成立.

例 7 求幂函数的 $n$ 阶导数公式.

解 设 $y=x^{\prime \prime} \quad(\mu$ 是任意常数)，那么

$$
\begin{aligned}
& y^{\prime}=\mu x^{\mu-1}, \\
& y^{\prime \prime}=\mu(\mu-1) x^{\mu-2}, \\
& y^{\prime \prime}=\mu(\mu-1)(\mu-2) x^{\mu-3}, \\
& y^{(4)}=\mu(\mu-1)(\mu-2)(\mu-3) x^{\mu-4},
\end{aligned}
$$

一般的, 可得

即

$$
y^{(\mu)}=\mu(\mu-1)(\mu-2) \cdots(\mu-n+1) x^{\mu-n},
$$

当 $\mu=n$ 时, 得到

$$
\left(x^{\mu}\right)^{(n)}=\mu(\mu-1)(\mu-2) \cdots(\mu-n+1) x^{\prime \prime} " \text {. }
$$

$$
\left(x^{n}\right)^{(n)}=n(n-1)(n-2) \cdots 3 \cdot 2 \cdot 1=n !,
$$

而

$$
\left(x^{n}\right)^{(n+1)}=0 \text {. }
$$

如果函数 $u=u(x)$ 及 $v=v(x)$ 都在点 $x$ 处具有 $n$ 阶导数, 那么显然 $u(x)$ $+v(x)$ 及 $u(x)-v(x)$ 也在点 $x$ 处具有 $n$ 阶导数, 且

$$
(u \pm v)^{(n)}=u^{(n)} \pm v^{(n)} \text {. }
$$

但乘积 $u(x) \cdot v(x)$ 的 $n$ 阶导数并不如此简单. 由

$$
(u v)^{\prime}=u^{\prime} v+u v^{\prime}
$$

首先得出

$$
\begin{gathered}
(u v)^{\prime \prime}=u^{\prime \prime} v+2 u^{\prime} v^{\prime}+u v^{\prime \prime}, \\
(u v)^{\prime \prime \prime}=u^{\prime \prime \prime} v+3 u^{\prime \prime} v^{\prime}+3 u^{\prime} v^{\prime \prime}+u v^{\prime \prime \prime} .
\end{gathered}
$$

用数学归纳法可以证明

$$
\begin{aligned}
(u v)^{(n)}= & u^{(n)} v+n u^{(n-1)} v^{\prime}+\frac{n(n-1)}{2 !} u^{(n-2)} v^{\prime \prime}+\cdots \\
& +\frac{n(n-1) \cdots(n-k+1)}{k !} u^{(n-k)} v^{(k)}+\cdots+u v^{(n)} .
\end{aligned}
$$

上式称为莱布尼茨 (Leibniz) 公式. 这公式可以这样记忆: 把 $(u+v)^{n}$ 按二项式 定理展开写成

$$
(u+v)^{n}=u^{n} v^{0}+n u^{n-1} v^{1}+\frac{n(n-1)}{2 !} u^{n-2} v^{2}+\cdots+u^{0} v^{n},
$$

即

$$
(u+v)^{n}=\sum_{k=0}^{n} \mathrm{C}_{n}^{k} u^{n-k} v^{k \Phi},
$$

然后把 $k$ 次幂换成 $k$ 阶导数 (零阶导数理解为函数本身), 再把左端的 $u+v$ 换 成 $u v$, 这样就得到莱布尼茨公式

$$
(u v)^{(n)}=\sum_{k=0}^{n} \mathrm{C}_{n}^{k} u^{(n-k)} v^{(k)} .
$$

例 $8 y=x^{2} \mathrm{e}^{2 x}$, 求 $y^{(20)}$.

解 设 $u=\mathrm{e}^{2 \cdot x}, v=x^{2}$, 则

$$
\begin{gathered}
u^{(k)}=2^{k} \mathrm{e}^{2 . x}(k=1,2, \cdots, 20), \\
v^{\prime}=2 x, \quad v^{\prime \prime}=2, \quad v^{(k)}=0(k=3,4, \cdots, 20),
\end{gathered}
$$

代入莱布尼茨公式, 得

$$
\begin{aligned}
y^{(211)} & =\left(x^{2} \mathrm{e}^{2 x}\right)^{(211)} \\
& =2^{20} \mathrm{e}^{2 x} \cdot x^{2}+20 \cdot 2^{19} \mathrm{e}^{2 x} \cdot 2 x+\frac{20 \cdot 19}{2 !} 2^{18} \mathrm{e}^{2 x} \cdot 2 \\
& =2^{20} \mathrm{e}^{2 x}\left(x^{2}+20 x+95\right) .
\end{aligned}
$$

(1) 记号 $\sum$ 表示对同一类型诸顶求和。例如、 $\sum_{k=1}^{n} \mathrm{C}_{n}^{k} u^{n-k} v^{k}$ 表示在 $\mathrm{C}_{n}^{k} u^{n-k} v^{k}$ 中依次令 $k=0,1, \cdots$, $n$, 然后对这样得到的 $n+1$ 顶求和.

## 习 题 2-3

1. 求下列函数的二阶导数:
(1) $y=2 x^{2}+\ln x$;
(2) $y=\mathrm{e}^{2 . r-1}$;
(3) $y=x \cos x$;
(4) $y=\mathrm{e}^{-t} \sin t$;
(5) $y=\sqrt{a^{2}-x^{2}}$;
(6) $y=\ln \left(1-x^{2}\right)$;
(7) $y=\tan x$;
(8) $y=\frac{1}{x^{3}+1}$;
(9) $y=\left(1+x^{2}\right) \arctan x$;
(10) $y=\frac{\mathrm{e}^{\prime}}{x}$;
(11) $y=x \mathrm{e}^{x^{2}}$;
(12) $y=\ln \left(x+\sqrt{1+x^{2}}\right)$.
2. 设 $f(x)=(x+10)^{6}, f^{m}(2)=$ ?
3. 设 $f^{\prime \prime}(x)$ 存在, 求下列函数的二阶导数 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$ :
(1) $y=f\left(x^{2}\right)$;
(2) $y=\ln [f(x)]$.
4. 试从 $\frac{\mathrm{d} x}{\mathrm{~d} y}=\frac{1}{y^{\prime}}$ 导出 :
(1) $\frac{\mathrm{d}^{2} x}{\mathrm{~d} y^{2}}=-\frac{y^{\prime \prime}}{\left(y^{\prime}\right)^{3}}$;
(2) $\frac{\mathrm{d}^{3} x}{\mathrm{~d} y^{3}}=\frac{3\left(y^{\prime \prime}\right)^{2}-y^{\prime} y^{\prime \prime}}{\left(y^{\prime}\right)^{5}}$.
5. 已知物体的运动规律为 $s=A \sin \omega t$ ( $A 、 \omega$ 是常数), 求物体运动的加速度, 并验证:

$$
\frac{\mathrm{d}^{2} s}{\mathrm{~d} t^{2}}+\omega^{2} s=0
$$

6. 密度大的隄星进人大气层时, 当它地心为 $s$ 千米时的速度与 $\sqrt{s}$ 成反比. 试证䧑星的 加速度与 $s^{2}$ 成反比.
7. 假设质点沿 $x$ 轴运动的速度为 $\frac{\mathrm{d} x}{\mathrm{~d} t}=f(x)$, 试求质点运动的加速度.
8. 验证函数 $y=C_{1} \mathrm{e}^{\lambda r}+C_{2} \mathrm{e}^{-\lambda r}\left(\lambda, C_{1}, C_{2}\right.$ 是常数 $)$ 满足关系式

$$
y^{\prime \prime}-\lambda^{2} y=0 \text {. }
$$

9. 验证函数 $y=\mathrm{e}^{\mathrm{d}} \sin x$ 满足关系式

$$
y^{\prime \prime}-2 y^{\prime}+2 y=0 \text {. }
$$

10. 求下列函数所指定的阶的导数:
(1) $y=\mathrm{e}^{\mathrm{e}} \cos x$, 求 $y^{(+)}$;
(2) $y=x^{2} \sin 2 x$, 求 $y^{(5 n)}$.

11. 求下列函数的n阶导数的一般表达式:
(1) $y=x^{n}+a_{1} x^{n-1}+a_{2} x^{n-2}+\cdots+a_{n-1} x+a_{n} \quad\left(a_{1}, a_{2}, \cdots, a_{n}\right.$ 都是常数)；
(2) $y=\sin ^{2} x$;
(3) $y=x \ln x$;
(4) $y=x \mathrm{e}^{\prime}$.

12.  求函数 $f(x)=x^{2} \ln (1+x)$ 在 $x=0$ 处的 $n$ 阶导数 $f^{(n)}(0)(n \geqslant 3)$.

## 第四节 隐函数及由参数方程所确定的

## 函数的导数 相关变化率

## 一、隐函数的导数

函数 $y=f(x)$ 表示两个变量 $y$ 与 $x$ 之间的对应关系, 这种对应关系可以用 各种不同方式表达. 前面我们遇到的函数,例如 $y=\sin x, y=\ln x+\sqrt{1-x^{2}}$ 等, 这种函数表达方式的特点是: 等号左端是因变量的符号, 而右端是含有自变量的 式子, 当自变量取定义域内任一值时, 由这式子能确定对应的函数值. 用这种方 式表达的函数叫做显函数. 有些函数的表达方式却不是这样,例如,方程

$$
x+y^{3}-1=0
$$

表示一个函数, 因为当变量 $x$ 在 $(-\infty,+\infty)$ 内取值时, 变量 $y$ 有确定的值与之 对应. 例如, 当 $x=0$ 时, $y=1$; 当 $x=-1$ 时, $y=\sqrt[3]{2}$, 等等. 这样的函数称为隐函 数.

一般的, 如果变量 $x$ 和 $y$ 满足一个方程 $F(x, y)=0$, 在一定条件下, 当 $x$ 取 某区间内的任一值时, 相应的总有满足这方程的唯一的 $y$ 值存在, 那么就说方 程 $F(x, y)=0$ 在该区间内确定了一个隐函数.

把一个隐函数化成显函数, 叫做隐函数的显化. 例如从方程 $x+y^{3}-1=0$ 解 出 $y=\sqrt[3]{1-x}$, 就把隐函数化成了显函数. 隐函数的显化有时是有困难的, 甚至 是不可能的. 但在实际问题中, 有时需要计算隐函数的导数, 因此, 我们希望有一 种方法, 不管隐函数能否显化, 都能直接由方程算出它所确定的隐函数的导数 来.下面通过具体例子来说明这种方法.

例 1 求由方程 $\mathrm{e}^{y}+x y-\mathrm{e}=0$ 所确定的隐函数的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.

解 我们把方程两边分别对 $x$ 求导数(1), 注意 $y=y(x)$. 方程左边对 $x$ 求 导得

$$
\frac{\mathrm{d}}{\mathrm{d} x}\left(\mathrm{e}^{y}+x y-\mathrm{e}\right)=\mathrm{e}^{y} \frac{\mathrm{d} y}{\mathrm{~d} x}+y+x \frac{\mathrm{d} y}{\mathrm{~d} x},
$$

方程右边对 $x$ 求导得

$$
(0)^{\prime}=0
$$

(1) 假设方程 $F(x, y)=0$ 确定一个函数 $y=y(x)$ ，把 $y=y(x)$ 代入方程侹得桓等式 $F[x, y(x)] \equiv 0$. 因此，这组说的方程两边对 $\boldsymbol{x}$ 求导，是指恒等式两边对 $\boldsymbol{x}$ 求导: 由于等式两边对 $x$ 的导数相等, 所以

$$
\mathrm{e}^{y} \frac{\mathrm{d} y}{\mathrm{~d} x}+y+x \frac{\mathrm{d} y}{\mathrm{~d} x}=0,
$$

从而

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=-\frac{y}{x+\mathrm{e}^{y}} \quad\left(x+\mathrm{e}^{y} \neq 0\right) .
$$

在这个结果中,分式中的 $y=y(x)$ 是由方程 $\mathrm{e}^{y}+x y-\mathrm{e}=0$ 所确定的隐函数.

例 2 求由方程 $y^{5}+2 y-x-3 x^{7}=0$ 所确定的隐函数在 $x=0$ 处的导数 $\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{x=0}$.

解 把方程两边分别对 $x$ 求导, 由于方程两边的导数相等, 所以

$$
5 y^{4} \frac{\mathrm{d} y}{\mathrm{~d} x}+2 \frac{\mathrm{d} y}{\mathrm{~d} x}-1-21 x^{\prime \prime}=0 \text {. }
$$

由此得

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1+21 x^{6}}{5 y^{4}+2} .
$$

因为当 $x=0$ 时, 从原方程得 $y=0$, 所以

$$
\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{x=0}=\frac{1}{2} \text {. }
$$

例 3 求椭圆 $\frac{x^{2}}{16}+\frac{y^{2}}{9}=1$ 在点 $\left(2, \frac{3}{2} \sqrt{3}\right)$ 处的切线方程 (图2-6).

解 由导数的几何意义知道,所求切线的斜率 为

$$
k=\left.y^{\prime}\right|_{x=2} .
$$

椭圆方程的两边分别对 $x$ 求导,有

$$
\frac{x}{8}+\frac{2}{9} y \cdot \frac{\mathrm{d} y}{\mathrm{~d} x}=0 \text {. }
$$

从而

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=-\frac{9 x}{16 y} \text {. }
$$

当 $x=2$ 时, $y=\frac{3}{2} \sqrt{3}$, 代入上式得

$$
\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{. x=2}=-\frac{\sqrt{3}}{4} .
$$

于是所求的切线方程为

$$
y-\frac{3}{2} \sqrt{3}=-\frac{\sqrt{3}}{4}(x-2),
$$

即

$$
\sqrt{3} x+4 y-8 \sqrt{3}=0 \text {. }
$$

例 4 求由方程 $x-y+\frac{1}{2} \sin y=0$ 所确定的隐函数的二阶导数 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$.

解 应用隐函数的求导方法, 得

于是

$$
1-\frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{1}{2} \cos y \cdot \frac{\mathrm{d} y}{\mathrm{~d} x}=0,
$$

上式两边再对 $x$ 求导, 得

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{2}{2-\cos y} \text {. }
$$

$$
\frac{d^{2} y}{d x^{2}}=\frac{-2 \sin y \frac{d y}{d x}}{(2-\cos y)^{2}}=\frac{-4 \sin y}{(2-\cos y)^{3}} .
$$

上式右端分式中的 $y=y(x)$ 是由方程 $x-y+\frac{1}{2} \sin y=0$ 所确定的隐函数.

在某些场合, 利用所谓对数求导法求导数比用通常的方法简便些. 这种方法 是先在 $y=f(x)$ 的两边取对数, 然后再求出 $y$ 的导数. 我们通过下面的例子来 说明这种方法.

例 5 求 $y=x^{\text {in } x}(x>0)$ 的导数.

解 这函数是幂指函数. 为了求这函数的导数, 可以先在两边取对数, 得

$$
\ln y=\sin x \cdot \ln x ;
$$

上式两边对 $x$ 求导, 注意到 $y=y(x)$, 得

$$
\frac{1}{y} y^{\prime}=\cos x \cdot \ln x+\sin x \cdot \frac{1}{x},
$$

于是 $y^{\prime}=y\left(\cos x \cdot \ln x+\frac{\sin x}{x}\right)=x^{\sin \cdot r}\left(\cos x \cdot \ln x+\frac{\sin x}{x}\right)$.

对于一般形式的幂指函数

$$
y=u^{\prime \prime}(u>0),
$$

如果 $u=u(x) 、 v=v(x)$ 都可导, 则可像例 5 那样利用对数求导法求出幂指函 数 (1) 的导数,也可把幂指函数 (1) 表示为

这样,便可直接求得

$$
y=\mathrm{e}^{\mathrm{vth} \text { " } . ~}
$$

$$
\begin{aligned}
y^{\prime} & =\mathrm{e}^{v \ln u}\left(v^{\prime} \cdot \ln u+v \cdot \frac{u^{\prime}}{u}\right) \\
& =u^{\prime}\left(v^{\prime} \cdot \ln u+\frac{v u^{\prime}}{u}\right) .
\end{aligned}
$$

例 6 求 $y=\sqrt{\frac{(x-1)(x-2)}{(x-3)(x-4)}}$ 的导数.

解 先在两边取对数(假定 $x>4$ ), 得

$$
\ln y=\frac{1}{2}[\ln (x-1)+\ln (x-2)-\ln (x-3)-\ln (x-4)],
$$

上式两边对 $x$ 求导, 注意到 $y=y(x)$, 得

$$
\frac{1}{y} y^{\prime}=\frac{1}{2}\left(\frac{1}{x-1}+\frac{1}{x-2}-\frac{1}{x-3}-\frac{1}{x-4}\right) \text {, }
$$

于是 $\quad y^{\prime}=\frac{y}{2}\left(\frac{1}{x-1}+\frac{1}{x-2}-\frac{1}{x-3}-\frac{1}{x-4}\right)$.

当 $2<x<3$ 时, $y=\sqrt{\frac{(x-1)(x-2)}{(3-x)(4-x)}}$;

用同样的方法可得与上面相同的结果.

## 二、由参数方程所确定的函数的导数

研究物体运动的轨迹时, 常遇到参数方程. 例如, 研究抛射体的运动问题时, 如果空气阻力忽略不计, 则抛射体的运动轨迹可表示为

$$
\left\{\begin{array}{l}
x=v_{1} t, \\
y=v_{2} t-\frac{1}{2} g t^{2},
\end{array}\right.
$$

其中 $v_{1} 、 v_{2}$ 分别是抛射体初速度的水平、铅直分量, $g$ 是重力加速度, $t$ 是飞行 时间, $x$ 和 $y$ 分别是飞行中抛射体在铅直平面上的位置的横坐标和纵坐标 (图 $2-7)$.

在 (2) 式中, $x, y$ 都与 $t$ 存在函数关 系. 如果把对应于同一个 $t$ 值的 $y$ 与 $x$ 的 值看做是对应的,这样就得到 $y$ 与 $x$ 之间 的函数关系. 消去(2)中的参数 $t$, 有

$$
y=\frac{v_{2}}{v_{1}} x-\frac{g}{2 v_{1}^{2}} x^{2} .
$$

这是因变量 $y$ 与自变量 $x$ 直接联系的式子,也是参数方程 (2) 所确定的函数的 显式表示。

一般的,若参数方程

$$
\left\{\begin{array}{l}
x=\varphi(t), \\
y=\psi(t)
\end{array}\right.
$$

确定 $y$ 与 $x$ 间的函数关系, 则称此函数关系所表达的函数为由参数方程 (3) 所 确定的函数. 在实际问题中,需要计算由参数方程 (3) 所确定的函数的导数. 但从 (3)中消 去参数 $t$ 有时会有困难. 因此, 我们希望有一种方法能直接由参数方程 (3) 算出 它所确定的函数的导数来.下面就来讨论由参数方程 (3) 所确定的函数的求导方 法。

在(3)式中, 如果函数 $x=\varphi(t)$ 具有单调连续反函数 $t=\varphi^{-1}(x)$, 且此反函 数能与函数 $y=\psi(t)$ 构成复合函数, 那么由参数方程 (3) 所确定的函数可以看 成是由函数 $y=\psi(t) 、 t=\varphi^{-1}(x)$ 复合而成的函数 $y=\psi\left[\varphi^{-1}(x)\right]$. 现在, 要计 算这个复合函数的导数. 为此再假定函数 $x=\varphi(t) 、 y=\psi(t)$ 都可导, 而且 $\varphi^{\prime}(t) \neq 0$. 于是根据复合函数的求导法则与反函数的求导法则, 就有

$$
\begin{gathered}
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} t} \cdot \frac{\mathrm{d} t}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} t} \cdot \frac{1}{\frac{\mathrm{d} x}{\mathrm{~d} t}}=\frac{\psi^{\prime}(t)}{\varphi^{\prime}(t)}, \\
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\psi^{\prime}(t)}{\varphi^{\prime}(t)} .
\end{gathered}
$$

上式也可写成

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\frac{\mathrm{d} y}{\mathrm{~d} t}}{\frac{\mathrm{d} x}{\mathrm{~d} t}} .
$$

(4) 式就是由参数方程 (3) 所确定的 $x$ 的函数的导数公式(1).

如果 $x=\varphi(t) 、 y=\psi(t)$ 还是二阶可导的, 那么从(4)式又可得到函数的二 阶导数公式

$$
\begin{aligned}
\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}} & =\frac{\mathrm{d}}{\mathrm{d} x}\left(\frac{\mathrm{d} y}{\mathrm{~d} x}\right)=\frac{\mathrm{d}}{\mathrm{d} t}\left(\frac{\psi^{\prime}(t)}{\varphi^{\prime}(t)}\right) \cdot \frac{\mathrm{d} t}{\mathrm{~d} x} \\
& =\frac{\psi^{\prime \prime}(t) \varphi^{\prime}(t)-\psi^{\prime}(t) \varphi^{\prime \prime}(t)}{\varphi^{\prime 2}(t)} \cdot \frac{1}{\varphi^{\prime}(t)}, \\
\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}} & =\frac{\psi^{\prime \prime}(t) \varphi^{\prime}(t)-\psi^{\prime}(t) \varphi^{\prime \prime}(t)}{\varphi^{\prime 3}(t)} .
\end{aligned}
$$

即

例 7 已知椭圆的参数方程为

$$
\left\{\begin{array}{l}
x=a \cos t, \\
y=b \sin t
\end{array}\right.
$$

$$
\left\{\begin{array}{l}
x=\varphi(t), \\
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\psi^{\prime}(t)}{\varphi^{\prime}(t)},
\end{array}\right.
$$

求椭圆在 $t=\frac{\pi}{4}$ 相应的点处的切线方程(图 2-8).

解 当 $t=\frac{\pi}{4}$ 时, 椭圆上的相应点 $M_{0}$ 的坐标是

$$
\begin{aligned}
& x_{0}=a \cos \frac{\pi}{4}=\frac{a \sqrt{2}}{2}, \\
& y_{0}=b \sin \frac{\pi}{4}=\frac{b \sqrt{2}}{2} .
\end{aligned}
$$

曲线在点 $M_{0}$ 的切线斜率为

$$
\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{t=\frac{\pi}{4}}=\left.\frac{(b \sin t)^{\prime}}{(a \cos t)^{\prime}}\right|_{t=\frac{\pi}{4}}=\left.\frac{b \cos t}{-a \sin t}\right|_{t=\frac{\pi}{4}}=-\frac{b}{a} .
$$

代入点斜式方程, 即得椭圆在点 $M_{0}$ 处的切线方程

$$
y-\frac{b \sqrt{2}}{2}=-\frac{b}{a}\left(x-\frac{a \sqrt{2}}{2}\right) \text {. }
$$

化简后得

$$
b x+a y-\sqrt{2} a b=0 .
$$

例 8 已知抛射体的运动轨迹的参数方程为

$$
\left\{\begin{array}{l}
x=v_{1} t, \\
y=v_{2} t-\frac{1}{2} g t^{2},
\end{array}\right.
$$

求抛射体在时刻 $t$ 的运动速度的大小和方向.

解 先求速度的大小.

由于速度的水平分量为

$$
\frac{\mathrm{d} x}{\mathrm{~d} t}=v_{1},
$$

铅直分量为

$$
\frac{\mathrm{d} y}{\mathrm{~d} t}=v_{2}-g t,
$$

所以抛射体运动速度的大小为

$$
v=\sqrt{\left(\frac{\mathrm{d} x}{\mathrm{~d} t}\right)^{2}+\left(\frac{\mathrm{d} y}{\mathrm{~d} t}\right)^{2}}=\sqrt{v_{1}^{2}+\left(v_{2}-g t\right)^{2}} .
$$

再求速度的方向,也就是轨迹的切线方向.

设 $\alpha$ 是切线的倾角, 则根据导数的几何意义,得

$$
\tan \alpha=\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\frac{\mathrm{d} y}{\mathrm{~d} t}}{\frac{\mathrm{d} x}{\mathrm{~d} t}}=\frac{v_{2}-g t}{v_{1}} .
$$

所以，在抛射体刚射出 (即 $t=0$ ) 时,

$$
\left.\tan \alpha\right|_{t<0}=\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{t=0}=\frac{v_{2}}{v_{1}} ;
$$

当 $t=\frac{v_{2}}{g}$ 时,

$$
\left.\tan \alpha\right|_{1=\frac{v_{2}}{k}}=\left.\frac{\mathrm{d} y}{\mathrm{~d} x}\right|_{1:=\frac{p_{2}}{\mathrm{~g}}}=0,
$$

这时, 运动方向是水平的, 即抛射体达到最高点(图 2-7).

例 9 计算由摆线 (图 2-9)的参数方程

$$
\left\{\begin{array}{l}
x=a(t-\sin t), \\
y=a(1-\cos t)
\end{array}\right.
$$

所确定的函数 $y=y(x)$ 的二阶导数.

解 $\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\frac{\mathrm{d} y}{\mathrm{~d} t}}{\frac{\mathrm{d} x}{\mathrm{~d} t}}=\frac{a \sin t}{a(1-\cos t)}=\frac{\sin t}{1-\cos t}=\cot \frac{t}{2}(t \neq 2 n \pi, n \in \mathbf{Z})$.

$$
\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}=\frac{\mathrm{d}}{\mathrm{d} t}\left(\cot \frac{t}{2}\right) \cdot \frac{1}{\frac{\mathrm{d} x}{\mathrm{~d} t}}
$$

$$
=-\frac{1}{2 \sin ^{2} \frac{t}{2}} \cdot \frac{1}{a(1-\cos t)}=-\frac{1}{a(1-\cos t)^{2}}
$$

$(t \neq 2 n \pi, n \in \mathbf{Z})$.

## 三、相关变化率

设 $x=x(t)$ 及 $y=y(t)$ 都是可导函数, 而变量 $x$ 与 $y$ 间存在某种关系, 从 而变化率 $\frac{\mathrm{d} x}{\mathrm{~d} t}$ 与 $\frac{\mathrm{d} y}{\mathrm{~d} t}$ 间也存在一定关系. 这两个相互依赖的变化率称为相关恋化 率. 相关变化率问题就是破究这两个变化率之间的关系, 以便从其中一个变化率 求出另一个变化率.

例 10 一气球从离开观察员 $500 \mathrm{~m}$ 处离地面铅直上升, 当气球高度为 $500 \mathrm{~m}$ 时, 其速率为 $140 \mathrm{~m} / \mathrm{min}$ (分). 求此时观察员视线的仰角增加的速率是多 少?

解 设气球上升 $t \mathrm{~s}$ (秒)后, 其高度为, $h$, 观察员视线的仰角为 $\alpha$, 则

$$
\tan \alpha=\frac{h}{500},
$$

其中 $\alpha$ 及 $h$ 都与 $t$ 存在可导的函数关系.上式两边对 $t$ 求导, 得

$$
\sec ^{2} \alpha \cdot \frac{\mathrm{d} \alpha}{\mathrm{d} t}=\frac{1}{500} \cdot \frac{\mathrm{d} h}{\mathrm{~d} t} .
$$

由已知条件, 存在 $t_{11}$, 使 $\left.h\right|_{1=t_{11}}=500 \mathrm{~m},\left.\frac{\mathrm{d} h}{\mathrm{~d} t}\right|_{t=t_{0}}=140 \mathrm{~m} / \mathrm{min}$. 又 $\left.\tan \alpha\right|_{1 ; t_{11}}=1,\left.\sec ^{2} \alpha\right|_{t: t_{0}}=2$. 代入上式得

$$
\left.2 \frac{\mathrm{d} \alpha}{\mathrm{d} t}\right|_{t=t_{0}}=\frac{1}{500} \cdot 140
$$

所以

$$
\left.\frac{\mathrm{d} \alpha}{\mathrm{d} t}\right|_{,=t_{\mathrm{II}}}=\frac{70}{500}=0.14(\mathrm{rad}(\text { 弧度 }) / \mathrm{min}) .
$$

即此时观察员视线的仰角增加的速率是 $0.14 \mathrm{rad} / \mathrm{min}$.

## 习 题 2-4

1. 求由下列方程所确定的隐函数的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ :
(1) $y^{2}-2 x y+9=0$;
(2) $x^{3}+y^{3}-3 a x y=0$;
(3) $x y=\mathrm{e}^{x+y}$;
(4) $y=1-x \mathrm{e}^{y}$. 2. 求曲线 $x^{\frac{2}{3}}+y^{\frac{2}{3}}=a^{\frac{2}{3}}$ 在点 $\left(\frac{\sqrt{2}}{4} a, \frac{\sqrt{2}}{4} a\right)$ 处的切线方程和法线方程.
2. 求由下列方程所确定的隐函数的二阶导数 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$ :
(1) $x^{2}-y^{2}=1$;
(2) $b^{2} x^{2}+a^{2} y^{2}=a^{2} b^{2}$;
(3) $y=\tan (x+y)$;
(4) $y=1+x \mathrm{e}^{y}$.
3. 用对数求导法求下列函数的导数:
(1) $y=\left(\frac{x}{1+x}\right)^{x}$;
(2) $y=\sqrt[5]{\frac{x-5}{\sqrt[5]{x^{2}+2}}}$;
(3) $y=\frac{\sqrt{x+2}(3-x)^{4}}{(x+1)^{5}}$;
(4) $y=\sqrt{x \sin x \sqrt{1-\mathrm{e}^{x}}}$.
4. 求下列参数方程所确定的函数的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ :
(1) $\left\{\begin{array}{l}x=a t^{2} \\ y=b t^{3}\end{array}\right.$
(2) $\left\{\begin{array}{l}x=\theta(1-\sin \theta), \\ y=\theta \cos \theta .\end{array}\right.$
5. 已知 $\left\{\begin{array}{l}x=\mathrm{e}^{t} \sin t \\ y=\mathrm{e}^{t} \cos t\end{array}\right.$ 求当 $t=\frac{\pi}{3}$ 时 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ 的值.
6. 写出下列曲线在所给参数值相应的点处的切线方程和法线方程:
(1) $\left\{\begin{array}{l}x=\sin t, \\ y=\cos 2 t,\end{array}\right.$ 在 $t=\frac{\pi}{4}$ 处;
(2) $\left\{\begin{array}{l}x=\frac{3 a t}{1+t^{2}}, \\ y=\frac{3 a t^{2}}{1+t^{2}},\end{array}\right.$ 在 $t=2$ 处.
7. 求下列参数方程所确定的函数的二阶导数 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$ :
(1) $\left\{\begin{array}{l}x=\frac{t^{2}}{2} \\ y=1-t ;\end{array}\right.$
(2) $\left\{\begin{array}{l}x=a \cos t \\ y=b \sin t\end{array}\right.$
(3) $\left\{\begin{array}{l}x=3 \mathrm{e}^{-1} \\ y=2 \mathrm{e}^{t} ;\end{array}\right.$
(4) $\left\{\begin{array}{l}x=f^{\prime}(t), \\ y=t f^{\prime}(t)-f(t) ;\end{array}\right.$ 设 $f^{\prime \prime}(t)$ 存在且不为零.

*9. 求下列参数方程所确定的函数的三阶导数 $\frac{\mathrm{d}^{3} y}{\mathrm{~d} x^{3}}$ :
(1) $\left\{\begin{array}{l}x=1-t^{2} \\ y=t-t^{3}\end{array}\right.$
(2) $\left\{\begin{array}{l}x=\ln \left(1+t^{2}\right), \\ y=t-\arctan t .\end{array}\right.$

10. 落在平静水面上的石头,产生同心波纹. 若最外一圈波半径的增大速率总是 $6 \mathrm{~m} / \mathrm{s}$, 问在 $2 \mathrm{~s}$ 末扰动水面面积增大的速率为多少?
11. 注水人深 $8 \mathrm{~m}$ 上顶直径 $8 \mathrm{~m}$ 的正圆锥形容器中, 其速率为 $4 \mathrm{~m}^{3} / \mathrm{min}$. 当水深为 $5 \mathrm{~m}$ 时,其表面上升的速率为多少?
12. 溶液自深 $18 \mathrm{~cm}$ 顶直径 $12 \mathrm{~cm}$ 的正圆锥形漏斗中漏人一直径为 $10 \mathrm{~cm}$ 的圆柱形筒 中. 开始时漏斗中盛满了溶液. 已知当溶液在漏斗中深为 $12 \mathrm{~cm}$ 时, 其表面下降的速率为 $1 \mathrm{~cm} / \mathrm{min}$. 问此时圆柱形筒中溶液表面上升的速率为多少?

## 第五节 函数的微分

## 一、微分的定义

先分析一个具体问题.一块正方形金属薄片受温度变化的影响, 其边长由 $x_{0}$ 变到 $x_{0}+\Delta x$ (图 2-10), 问此薄片的面积改变了多少?

设此薄片的边长为 $x$, 面积为 $A$, 则 $A$ 与 $x$ 存在函数关系: $A=x^{2}$. 薄片受 温度变化的影响时面积的改变量, 可以看成是当自变量 $x$ 自 $x_{0}$ 取得增量 $\Delta x$ 时, 函数 $A=x^{2}$ 相应的增量 $\Delta A$, 即

$$
\Delta A=\left(x_{0}+\Delta x\right)^{2}-x_{0}^{2}=2 x_{0} \Delta x+(\Delta x)^{2} .
$$

从上式可以看出, $\Delta A$ 分成两部分, 第一部分 $2 x_{11} \Delta x$ 是 $\Delta x$ 的线性函数, 即图中带有斜线的两 个矩形面积之和, 而第二部分 $(\Delta x)^{2}$ 在图中是带 有交叉斜线的小正方形的面积, 当 $\Delta x \rightarrow 0$ 时,第 二部分 $(\Delta x)^{2}$ 是比 $\Delta x$ 高阶的无穷小, 即 $(\Delta x)^{2}=$ $o(\Delta x)$. 由此可见, 如果边长改变很微小, 即 $|\Delta x|$ 很小时,面积的改变量 $\Delta A$ 可近似地用第一部分 来代替。

一般的, 如果函数 $y=f(x)$ 满足一定条件, 则 增量 $\Delta y$ 可表示为

$$
\Delta y=A \Delta x+o(\Delta x),
$$

其中 $A$ 是不依赖于 $\Delta x$ 的常数, 因此 $A \Delta x$ 是 $\Delta x$ 的线性函数, 且它与 $\Delta y$ 之差

$$
\Delta y-A \Delta x=o(\Delta x)
$$

是比 $\Delta x$ 高阶的无穷小. 所以, 当 $A \neq 0$, 且 $|\Delta x|$ 很小时, 我们就可以用 $\Delta x$ 的线 性函数 $A \Delta x$ 来近似代替 $\Delta y$.

定义 设函数 $y=f(x)$ 在某区间内有定义, $x_{0}$ 及 $x_{0}+\Delta x$ 在这区间内, 如 果增量

$$
\Delta y=f\left(x_{11}+\Delta x\right)-f\left(x_{0}\right)
$$

## 可表示为

$$
\Delta y=A \Delta x+o(\Delta x),
$$

其中 $A$ 是不依赖于 $\Delta x$ 的常数, 那么称函数 $y=f(x)$ 在点 $x_{0}$ 是可微的, 而 $A \Delta x$ 叫做函数 $y=f(x)$ 在点 $x_{0}$ 相应于自变量增量 $\Delta x$ 的微分, 记作 $\mathrm{d} y$, 即

$$
\mathrm{d} y=A \Delta x \text {. }
$$

下面讨论函数可微的条件. 设函数 $y=f(x)$ 在点 $x_{0}$ 可微, 则按定义有 (1) 式成立. (1)式两边除以 $\Delta x$,得

$$
\frac{\Delta y}{\Delta x}=A+\frac{o(\Delta x)}{\Delta x} .
$$

于是, 当 $\Delta x \rightarrow 0$ 时, 由上式就得到

$$
A=\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=f^{\prime}\left(x_{0}\right) .
$$

因此, 如果函数 $f(x)$ 在点 $x_{0}$ 可微, 则 $f(x)$ 在点 $x_{0}$ 也一定可导 (即 $f^{\prime}\left(x_{0}\right)$ 存 在), 且 $A=f^{\prime}\left(x_{0}\right)$.

反之, 如果 $y=f(x)$ 在点 $x_{0}$ 可导, 即

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=f^{\prime}\left(x_{.0}\right)
$$

存在, 根据极限与无穷小的关系(第一章第四节定理 1), 上式可写成

$$
\frac{\Delta y}{\Delta x}=f^{\prime}\left(x_{1}\right)+\alpha,
$$

其中 $\alpha \rightarrow 0$ (当 $\Delta x \rightarrow 0$ ). 由此又有

$$
\Delta y=f^{\prime}\left(x_{0}\right) \Delta x+\alpha \Delta x .
$$

因 $\alpha \Delta x=o(\Delta x)$, 且 $f^{\prime}\left(x_{0}\right)$ 不依赖于 $\Delta x$, 故上式相当于 (1) 式, 所以 $f(x)$ 在点 $x_{0}$ 也是可微的.

由此可见,函数 $f(x)$ 在点 $x_{1}$ 可微的充分必要条件是函数 $f(x)$ 在点 $x_{0}$ 可 导, 且当 $f(x)$ 在点 $x_{0}$ 可微时, 其微分一定是

$$
\mathrm{d} y=f^{\prime}\left(x_{0}\right) \Delta x .
$$

当 $f^{\prime}\left(x_{01}\right) \neq 0$ 时, 有

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\mathrm{~d} y}=\lim _{\Delta r^{\rightarrow} \rightarrow 0} \frac{\Delta y}{f^{\prime}\left(x_{10}\right) \Delta x}=\frac{1}{f^{\prime}\left(x_{11}\right)} \lim _{\Delta x \rightarrow 011} \frac{\Delta y}{\Delta x}=1 .
$$

从而,当 $\Delta x \rightarrow 0$ 时, $\Delta y$ 与 $\mathrm{d} y$ 是等价无穷小, 于是由第一章第七节定理 1 可知, 这时有

$$
\Delta y=\mathrm{d} y+o(\mathrm{~d} y),
$$

即 $\mathrm{d} y$ 是 $\Delta y$ 的主部 (1). 又由于 $\mathrm{d} y=f^{\prime}\left(x_{0}\right) \Delta x$ 是 $\Delta x$ 的线性函数, 所以在

$f^{\prime}\left(x_{0}\right) \neq 0$ 的条件下, 我们说 $\mathrm{d} y$ 是 $\Delta y$ 的线性主部 (当 $\Delta x \rightarrow 0$ ). 于是我们得到 结论: 在 $f^{\prime}\left(x_{0}\right) \neq 0$ 的条件下, 以微分 $\mathrm{d} y=f^{\prime}\left(x_{0}\right) \Delta x$ 近似代替增量 $\Delta y=$ $f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)$ 时, 其误差为 $o(\mathrm{~d} y)$. 因此, 在 $|\Delta x|$ 很小时, 有近似等式

$$
\Delta y \approx \mathrm{d} y \text {. }
$$

例 1 求函数 $y=x^{2}$ 在 $x=1$ 和 $x=3$ 处的微分.

解 函数 $y=x^{2}$ 在 $x=1$ 处的微分为

$$
\mathrm{d} y=\left.\left(x^{2}\right)^{\prime}\right|_{x=1} \Delta x=2 \Delta x ;
$$

在 $x=3$ 处的微分为

$$
\mathrm{d} y=\left(x^{2}\right)^{\prime} 1_{x=3} \Delta x=6 \Delta x .
$$

函数 $y=f(x)$ 在任意点 $x$ 的微分, 称为函数的微分, 记作 $\mathrm{d} y$ 或 $\mathrm{d} f(x)$, 即

$$
\mathrm{d} y=f^{\prime}(x) \Delta x \text {. }
$$

例如, 函数 $y=\cos x$ 的微分为

$$
\mathrm{d} y=(\cos x)^{\prime} \Delta x=-\sin x \Delta x ;
$$

函数 $y=\mathrm{e}^{x}$ 的微分为

$$
\mathrm{d} y=\left(\mathrm{e}^{\mathrm{r}}\right)^{\prime} \Delta x=\mathrm{e}^{r} \Delta x .
$$

显然, 函数的微分 $\mathrm{d} y=f^{\prime}(x) \Delta x$ 与 $x$ 和 $\Delta x$ 有关.

例 2 求函数 $y=x^{3}$ 当 $x=2, \Delta x=0.02$ 时的微分.

解 先求函数在任意点 $x$ 的微分

$$
\mathrm{d} y=\left(x^{3}\right)^{\prime} \Delta x=3 x^{2} \Delta x .
$$

再求函数当 $x=2, \Delta x=0.02$ 时的微分

$$
\left.\mathrm{d} y\right|_{\substack{x=2 \\ \Delta x=0.02}}=\left.3 x^{2} \Delta x\right|_{\substack{x=2 \\ \Delta x=0.02}}=3 \cdot 2^{2} \cdot 0.02=0.24 .
$$

通常把自变量 $x$ 的增量 $\Delta x$ 称为自变量的微分, 记作 $\mathrm{d} x$, 即 $\mathrm{d} x=\Delta x$. 于是 函数 $y=f(x)$ 的微分又可记作

$$
\text { 从而有 } \begin{aligned}
\mathrm{d} y & =f^{\prime}(x) \mathrm{d} x . \\
\frac{\mathrm{d} y}{\mathrm{~d} x} & =f^{\prime}(x) .
\end{aligned}
$$

这就是说, 函数的微分 $\mathrm{d} y$ 与自变量的微分 $\mathrm{d} x$ 之商等于该函数的导数. 因此, 导 数也叫做“微商”。

## 二、微分的几何意义

为了对微分有比较直观的了解, 我们来说明微分的几何意义.

在直角坐标系中, 函数 $y=f(x)$ 的图形是一条曲线. 对于某一固定的 $x_{n}$ 值, 曲线上有一个确定点 $M\left(x_{0}, y_{11}\right)$, 当自变量 $x$ 有微小增量 $\Delta x$ 时, 就得到曲 线上另一点 $N\left(x_{0}+\Delta x, y_{0}+\Delta y\right)$. 从图 2-11 可知:

$$
\begin{aligned}
& M Q=\Delta x, \\
& Q N=\Delta y .
\end{aligned}
$$

过点 $M$ 作曲线的切线 $M T$, 它的倾角为 $\alpha$, 则

$$
Q P=M Q \cdot \tan \alpha=\Delta x \cdot f^{\prime}\left(x_{0}\right),
$$

即

$$
\mathrm{d} y=Q P .
$$

由此可见, 对于可微函数 $y=f(x)$ 而言, 当 $\Delta y$ 是曲线 $y=f(x)$ 上的点的纵坐标的增量时, $\mathrm{d} y$ 就是曲线的切线上点的纵坐标的相应增量. 当 $|\Delta x|$ 很小时, $|\Delta y-\mathrm{d} y|$ 比 $|\Delta x|$ 小得多. 因此在点

国 $2-11$ $M$ 的邻近,我们可以用切线段来近似代替曲线段.在局部范围内用线性函数近 似代替非线性函数,在几何上就是局部用切线段近似代替曲线段,这在数学上称 为非线性函数的局部线性化, 这是微分学的基本思想方法之一. 这种思想方法在 自然科学和工程问题的研究中是经常采用的.

## 三、基本初等函数的微分公式与微分运算法则

从函数的微分的表达式

$$
\mathrm{d} y=f^{\prime}(x) \mathrm{d} x
$$

可以看出, 要计算函数的微分, 只要计算函数的导数, 再乘以自变冒的微分. 因 此, 可得如下的微分公式和微分运算法则.

## 1. 基本初等函数的微分公式

由基本初等函数的导数公式,可以直接写出基本初等函数的微分公式. 为了 便于对照,列表于下:

| 导 数 公 式 | 微 分 公 式 |
| :---: | :---: |
| $\left(x^{\prime \prime}\right)^{\prime}=\mu x^{\prime \prime-1}$ | $\mathrm{~d}\left(x^{\prime \prime}\right)=\mu x^{\prime \prime} \mathrm{d} x$ |
| $(\sin x)^{\prime}=\cos x$ | $\mathrm{~d}(\sin x)=\cos x \mathrm{~d} x$ |
| $(\cos x)^{\prime}=-\sin x$ | $\mathrm{~d}(\cos x)=-\sin x \mathrm{~d} x$ |
| $(\tan x)^{\prime}=\sec ^{2} x$ | $\mathrm{~d}(\tan x)=\sec ^{2} x \mathrm{~d} x$ |
| $(\cot x)^{\prime}=-\csc ^{2} x$ | $\mathrm{~d}(\cot x)=-\csc ^{2} x \mathrm{~d} x$ |
| $(\sec x)^{\prime}=\sec x \tan x$ | $\mathrm{~d}(\sec x)=\sec x \tan x \mathrm{~d} x$ |
| $(\csc x)^{\prime}=-\csc x \cot x$ | $\mathrm{~d}(\csc x)=-\csc x \cot x \mathrm{~d} x$ |

| 导 数 公式 | 微 分 公 式 |
| ---: | :---: |
| $\left(a^{x}\right)^{\prime}=a^{x} \ln a$ | $\mathrm{~d}\left(a^{x}\right)=a^{x} \ln a \mathrm{~d} x$ |
| $\left(\mathrm{e}^{\prime}\right)^{\prime}=\mathrm{e}^{x}$ | $\mathrm{~d}\left(\mathrm{e}^{x}\right)=\mathrm{e}^{x} \mathrm{~d} x$ |
| $\left(\log _{a} x\right)^{\prime}=\frac{1}{x \ln a}$ | $\mathrm{~d}\left(\log _{\mathrm{u}} x\right)=\frac{1}{x \ln a} \mathrm{~d} x$ |
| $(\ln x)^{\prime}=\frac{1}{x}$ | $\mathrm{~d}(\ln x)=\frac{1}{x} \mathrm{~d} x$ |
| $(\arcsin x)^{\prime}=\frac{1}{\sqrt{1-x^{2}}}$ | $\mathrm{~d}(\arcsin x)=\frac{1}{\sqrt{1-x^{2}} \mathrm{~d} x}$ |
| $(\arccos x)^{\prime}=-\frac{1}{\sqrt{1-x^{2}}}$ | $\mathrm{~d}(\arccos x)=-\frac{1}{\sqrt{1-x^{2}} \mathrm{~d} x}$ |
| $(\arctan x)^{\prime}=\frac{1}{1+x^{2}}$ | $\mathrm{~d}(\arctan x)=\frac{1}{1+x^{2}} \mathrm{~d} x$ |
| $(\operatorname{arccot} x)^{\prime}=-\frac{1}{1+x^{2}}$ | $\mathrm{~d}(\operatorname{arccot} x)=-\frac{1}{1+x^{2}} \mathrm{~d} x$ |

2. 函数和、差、积、商的微分法则

由函数和、差、积、商的求导法则, 可推得相应的微分法则. 为了便于对照, 列 成下表(表中 $u=u(x), v=v(x)$ 都可导).

| 函数和、差、积、商的求导法则 | 函数和、差、积、商的微分法则 |
| :---: | :---: |
| $(u \pm v)^{\prime}=u^{\prime} \pm v^{\prime}$ | $\mathrm{d}(u \pm v)=\mathrm{d} u \pm \mathrm{d} v$ |
| $(C u)^{\prime}=C u^{\prime}$ | $\mathrm{d}(C u)=C \mathrm{~d} u$ |
| $(u v)^{\prime}=u^{\prime} v+u v^{\prime}$ | $\mathrm{d}(u v)=v \mathrm{~d} u+u \mathrm{~d} v$ |
| $\left(\frac{u}{v}\right)^{\prime}=\frac{u^{\prime} v-u v^{\prime}}{v^{2}}(v \neq 0)$ | $\mathrm{d}\left(\frac{u}{v}\right)=\frac{v \mathrm{~d} u-u \mathrm{~d} v}{v^{2}} \quad(v \neq 0)$ |

现在我们以乘积的微分法则为例加以证明.

根据函数微分的表达式, 有

$$
\mathrm{d}(u v)=(u v)^{\prime} \mathrm{d} x .
$$

再根据乘积的求导法则, 有

$$
(u v)^{\prime}=u^{\prime} v+u v^{\prime} .
$$

于是

$$
\mathrm{d}(u v)=\left(u^{\prime} v+u v^{\prime}\right) \mathrm{d} x=u^{\prime} v \mathrm{~d} x+u v^{\prime} \mathrm{d} x \text {. }
$$

由于 $u^{\prime} \mathrm{d} x=\mathrm{d} u, v^{\prime} \mathrm{d} x=\mathrm{d} v$, 其他法则都可以用类似方法证明.

## 3. 复合函数的微分法则

与复合函数的求导法则相应的复合函数的微分法则可推导如下:

设 $y=f(u)$ 及 $u=g(x)$ 都可导, 则复合函数 $y=f[g(x)]$ 的微分为

$$
\mathrm{d} y=y^{\prime}{ }_{r} \mathrm{~d} x=f^{\prime}(u) g^{\prime}(x) \mathrm{d} x .
$$

由于 $g^{\prime}(x) \mathrm{d} x=\mathrm{d} u$, 所以, 复合函数 $y=f[g(x)]$ 的微分公式也可以写成

$$
\mathrm{d} y=f^{\prime}(u) \mathrm{d} u \text { 或 } \mathrm{d} y=y^{\prime}{ }^{\prime} \mathrm{d} u \text {. }
$$

由此可见,无论 $u$ 是自变量还是中间变量,微分形式 $\mathrm{d} y=f^{\prime}(u) \mathrm{d} u$ 保持不 变. 这一性质称为微分形式不变性. 这性质表示, 当变换自变量时, 微分形式 $\mathrm{d} y=f^{\prime}(u) \mathrm{d} u$ 并不改变.

例 $3 y=\sin (2 x+1)$, 求 $\mathrm{d} y$.

解 把 $2 x+1$ 看成中间变量 $u$,则

$$
\begin{aligned}
\mathrm{d} y & =\mathrm{d}(\sin u)=\cos u \mathrm{~d} u=\cos (2 x+1) \mathrm{d}(2 x+1) \\
& =\cos (2 x+1) \cdot 2 \mathrm{~d} x=2 \cos (2 x+1) \mathrm{d} x .
\end{aligned}
$$

在求复合函数的导数时, 可以不写出中间变量. 在求复合函数的微分时, 类 似地也可以不写出中间变量.下面我们用这种方法来求函数的微分.

例 $4 y=\ln \left(1+\mathrm{e}^{x^{2}}\right)$, 求 $\mathrm{d} y$.

解 $\mathrm{d} y=\mathrm{d}\left(\ln \left(1+\mathrm{e}^{x^{2}}\right)\right)=\frac{1}{1+\mathrm{e}^{x^{2}}} \mathrm{~d}\left(1+\mathrm{e}^{x^{2}}\right)=\frac{1}{1+\mathrm{e}^{x^{2}}} \cdot \mathrm{e}^{x^{2}} \mathrm{~d}\left(x^{2}\right)$

$$
=\frac{\mathrm{e}^{x^{2}}}{1+\mathrm{e}^{x^{2}}} \cdot 2 x \mathrm{~d} x=\frac{2 x \mathrm{e}^{x^{2}}}{1+\mathrm{e}^{x^{2}}} \mathrm{~d} x \text {. }
$$

例 $5 y=\mathrm{e}^{3-3 x} \cos x$, 求 $\mathrm{d} y$.

解 应用积的微分法则, 得

$$
\begin{aligned}
\mathrm{d} y & =\mathrm{d}\left(\mathrm{e}^{1-3 x} \cos x\right)=\cos x \mathrm{~d}\left(\mathrm{e}^{1-3 x}\right)+\mathrm{e}^{1-3 x} \mathrm{~d}(\cos x) \\
& =(\cos x) \mathrm{e}^{1-3 x}(-3 \mathrm{~d} x)+\mathrm{e}^{1-3 x}(-\sin x \mathrm{~d} x) \\
& =-\mathrm{e}^{1-3 x}(3 \cos x+\sin x) \mathrm{d} x .
\end{aligned}
$$

例 6 在下列等式左端的括号中填入适当的函数,使等式成立.
(1) $\mathrm{d}(\quad)=x \mathrm{~d} x$;
(2) $\mathrm{d}(\quad)=\cos \omega t \mathrm{~d} t$.

解 (1) 我们知道,

可见

$$
\begin{gathered}
\mathrm{d}\left(x^{2}\right)=2 x \mathrm{~d} x . \\
x \mathrm{~d} x=\frac{1}{2} \mathrm{~d}\left(x^{2}\right)=\mathrm{d}\left(\frac{x^{2}}{2}\right),
\end{gathered}
$$

$$
\mathrm{d}\left(\frac{x^{2}}{2}\right)=x \mathrm{~d} x
$$

一般的, 有

$$
\left.\mathrm{d}\left(\frac{x^{2}}{2}+C\right)=x \mathrm{~d} x \text { ( } C \text { 为任意常数 }\right) \text {. }
$$

（2）因为

$$
\mathrm{d}(\sin \omega t)=\omega \cos \omega t \mathrm{~d} t,
$$

可见

$$
\cos \omega t \mathrm{~d} t=\frac{1}{\omega} \mathrm{d}(\sin \omega t)=\mathrm{d}\left(\frac{1}{\omega} \sin \omega t\right),
$$

即

$$
\mathrm{d}\left(\frac{1}{\omega} \sin \omega t\right)=\cos \omega t \mathrm{~d} t .
$$

一般的, 有

$$
\mathrm{d}\left(\frac{1}{\omega} \sin \omega t+C\right)=\cos \omega t \mathrm{~d} t \text { ( } C \text { 为任意常数). }
$$

## 四、微分在近似计算中的应用

## 1. 函数的近似计算

在工程问题中, 经常会遏到一些复杂的计算公式. 如果直接用这些公式进行 计算, 那是很费力的. 利用微分往往可以把一些复杂的计算公式用简单的近似公 式来代替.

前面说过, 如果 $y=f(x)$ 在点 $x_{4}$ 处的导数 $f^{\prime}\left(x_{0}\right) \neq 0$, 且 $|\Delta x|$ 很小时, 我 们有

$$
\Delta y \approx \mathrm{d} y=f^{\prime}\left(x_{0}\right) \Delta x
$$

这个式子也可以写为

$$
\Delta y=f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right) \approx f^{\prime}\left(x_{0}\right) \Delta x,
$$

或

$$
f\left(x_{0}+\Delta x\right) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right) \Delta x .
$$

在 (5) 式中令 $x=x_{1}+\Delta x$, 即 $\Delta x=x-x_{0}$, 那么 (5) 式可改写为

$$
f(x) \approx f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right) \text {. }
$$

如果 $f\left(x_{11}\right)$ 与 $f^{\prime}\left(x_{11}\right)$ 都容易计算, 那么可利用 $(4)$ 式来近似计算 $\Delta y$, 利用 (5)式来近似计算 $f\left(x_{1}+\Delta x\right)$, 或利用 (6) 式来近似计算 $f(x)$. 这种近似计算的 实质就是用 $x$ 的线性函数 $f\left(x_{10}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)$ 来近似表达函数 $f(x)$. 从导 数的几何意义可知, 这也就是用曲线 $y=f(x)$ 在点 $\left(x_{0}, f\left(x_{0}\right)\right)$ 处的切线来近 似代替该曲线 (就切点邻近部分来说). 例 7 有一批半径为 $1 \mathrm{~cm}$ 的球, 为了提高球面的光洁度, 要铰上一层铜, 厚 度定为 $0.01 \mathrm{~cm}$. 估计一下每只球需用铜多少 $\mathrm{g}\left(\right.$ 铜的密度是 $\left.8.9 \mathrm{~g} / \mathrm{cm}^{3}\right)$ ?

解 先求出镀层的体积, 再乘上密度就得到每只球需用铜的质量.

因为镀层的体积等于两个球体体积之差, 所以它就是球体体积 $V=\frac{4}{3} \pi R^{3}$ 当 $R$ 自 $R_{0}$ 取得增量 $\Delta R$ 时的增量 $\Delta V$. 我们求 $V$ 对 $R$ 的导数

$$
\left.V^{\prime}\right|_{R=R_{0}}=\left.\left(\frac{4}{3} \pi R^{3}\right)^{\prime}\right|_{R=R_{0}}=4 \pi R_{0}^{2},
$$

由 (4)式得

$$
\Delta V \approx 4 \pi R_{0}^{2} \Delta R \text {. }
$$

将 $R_{0}=1 、 \Delta R=0.01$ 代入上式, 得

$$
\Delta V \approx 4 \times 3.14 \times 1^{2} \times 0.01 \approx 0.13\left(\mathrm{~cm}^{3}\right) .
$$

于是镀每只球需用的铜约为

$$
0.13 \times 8.9 \approx 1.16(\mathrm{~g}) \text {. }
$$

例 8 利用微分计算 $\sin 30^{\circ} 30^{\prime}$ 的近似值.

解 把 $30^{\circ} 30^{\prime}$ 化为弧度, 得

$$
30^{\circ} 30^{\prime}=\frac{\pi}{6}+\frac{\pi}{360} .
$$

由于所求的是正弦函数的值, 故设 $f(x)=\sin x$. 此时 $f^{\prime}(x)=\cos x$. 如果 取 $x_{0}=\frac{\pi}{6}$, 则 $f\left(\frac{\pi}{6}\right)=\sin \frac{\pi}{6}=\frac{1}{2}$ 与 $f^{\prime}\left(\frac{\pi}{6}\right)=\cos \frac{\pi}{6}=\frac{\sqrt{3}}{2}$ 都容易计算, 并且 $\Delta x=\frac{\pi}{360}$ 比较小. 应用 $(5)$ 式便得

$$
\begin{aligned}
\sin 30^{\circ} 30^{\prime} & =\sin \left(\frac{\pi}{6}+\frac{\pi}{360}\right) \approx \sin \frac{\pi}{6}+\cos \frac{\pi}{6} \cdot \frac{\pi}{360} \\
& =\frac{1}{2}+\frac{\sqrt{3}}{2} \cdot \frac{\pi}{360} \approx 0.5000+0.0076 \\
& =0.5076 .
\end{aligned}
$$

下面我们来推导一些常用的近似公式. 为此, 在 (6) 式中取 $x_{0}=0$,于是得

$$
f(x) \approx f(0)+f^{\prime}(0) x .
$$

应用(7)式可以推得以下几个在工程上常用的近似公式(下面都假定 $|x|$ 是 较小的数值):

(i) $\sqrt[n]{1+x} \approx 1+\frac{1}{n} x$;

(ii) $\sin x \approx x$ ( $x$ 用弧度作单位来表达);

(iii) $\tan x \approx x$ ( $x$ 用弧度作单位来表达);

(iv) $\mathrm{e}^{x} \approx 1+x$; (v) $\ln (1+x) \approx x$.

证（i）在第一章第七节例 1 中我们已经证明过这个近似公式. 在这里, 我 们利用微分证明. 取 $f(x)=\sqrt[n]{1+x}$, 那么 $f(0)=1, f^{\prime}(0)=\left.\frac{1}{n}(1+x)^{\frac{1}{n}-1}\right|_{x=0}$ $=\frac{1}{n}$, 代入(7)式便得

$$
\sqrt[n]{1+x} \approx 1+\frac{1}{n} x
$$

证 (ii) 取 $f(x)=\sin x$, 那么 $f(0)=0, f^{\prime}(0)=\left.\cos x\right|_{x=0}=1$, 代入(7)式 便得

$$
\sin x \approx x .
$$

其他几个近似公式可用类似方法证明,这里从略了.

例 9 计算 $\sqrt{1.05}$ 的近似值.

解

$$
\sqrt{1.05}=\sqrt{1+0.05},
$$

这里 $x=0.05$, 其值较小,利用近似公式 ( $\mathrm{i})$ ( $n=2$ 的情形), 便得

$$
\sqrt{1.05} \approx 1+\frac{1}{2}(0.05)=1.025 .
$$

如果直接开方, 可得

$$
\sqrt{1.05}=1.02470 .
$$

将两个结果比较一下, 可以看出, 用 1.025 作为 $\sqrt{1.05}$ 的近似值, 其误差不超过 0.001 , 这样的近似值在一般应用上已够精确了. 如果开方次数较高, 就更能体现 出用微分进行近似计算的优越性.

## 2. 误差估计

在生产实践中,经常要测量各种数据. 但是有的数据不易直接测量, 这时我 们就通过测量其他有关数据后，根据某种公式算出所要的数据. 例如,要计算圆 钢的截面积 $A$, 可先用卡尺测量圆钢截面的直径 $D$, 然后根据公式 $A=\frac{\pi}{4} D^{2}$ 算 出 $A$.

由于测墨仪器的精度、测量的条件和测量的方法等各种因素的影响,测得的 数据往往带有误差, 而根据带有误差的数据计算所得的结果也会有误差,我们把 它叫做间接测霓误差.

下面就讨论怎样利用微分来估计间接测量误差.

先说明什么叫绝对误差、什么叫相对误差.

如果某个柾的精确值为 $A$, 它的近似值为 $a$, 那么 $|A-a|$ 叫做 $a$ 的绝对误 差,而绝对误差与 $|a|$ 的比值 $\frac{|A-a|}{|a|}$ 叫做 $a$ 的相对误差.

在实际工作中, 某个量的精确值往往是无法知道的, 于是绝对误差和相对误 差也就无法求得. 但是根据测量仪器的精度等因素, 有时能够确定误差在某一个 范围内. 如果某个量的精确值是 $A$, 测得它的近似值是 $a$, 又知道它的误差不超 过 $\delta_{\Lambda}$, 即

$$
|A-a| \leqslant \delta_{\lambda},
$$

那么 $\delta_{\Lambda}$ 叫做测量 $A$ 的绝对误差限, 而 $\frac{\delta_{A}}{|a|}$ 叫做测量 $A$ 的相对误差限.

例 10 设测得圆钢截面的直径 $D=60.03 \mathrm{~mm}$, 测量 $D$ 的绝对误差限 $\delta_{D}=$ $0.05 \mathrm{~mm}$. 利用公式

$$
A=\frac{\pi}{4} D^{2}
$$

计算圆钢的截面积时,试估计面积的误差.

解 我们把测量 $D$ 时所产生的误差当作自变量 $D$ 的增量 $\Delta D$, 那么, 利用 公式 $A=\frac{\pi}{4} D^{2}$ 来计算 $A$ 时所产生的误差就是函数 $A$ 的对应增量 $\Delta A$. 当 $|\Delta D|$ 很小时, 可以利用微分 $\mathrm{d} A$ 近似地代替增量 $\Delta A$, 即

$$
\Delta A \approx \mathrm{d} A=A^{\prime} \cdot \Delta D=\frac{\pi}{2} D \cdot \Delta D .
$$

由于 $D$ 的绝对误差限为 $\delta_{D}=0.05 \mathrm{~mm}$, 所以

$$
|\Delta D| \leqslant \delta_{D}=0.05 \text {, }
$$

而

$$
|\Delta A| \approx|\mathrm{d} A|=\frac{\pi}{2} D \cdot|\Delta D| \leqslant \frac{\pi}{2} D \cdot \delta_{\mathrm{D}},
$$

因此得出 $A$ 的绝对误差限约为

$$
\delta_{A}=\frac{\pi}{2} D \cdot \delta_{D}=\frac{\pi}{2} \times 60.03 \times 0.05 \approx 4.715\left(\mathrm{~mm}^{2}\right) ;
$$

$A$ 的相对误差限约为

$$
\frac{\delta_{A}}{A}=\frac{\frac{\pi}{2} D \cdot \delta_{D}}{\frac{\pi}{4} D^{2}}=2 \frac{\delta_{D}}{D}=2 \times \frac{0.05}{60.03} \approx 0.17 \% .
$$

一般的, 根据直接测量的 $x$ 值按公式 $y=f(x)$ 计算 $y$ 值时, 如果已知测量 $x$ 的绝对误差限是 $\delta_{a}$, 即

$$
|\Delta x| \leqslant \delta_{r},
$$

那么, 当 $y^{\prime} \neq 0$ 时, $y$ 的绝对误差

$$
|\Delta y| \approx|\mathrm{d} y|=\left|y^{\prime}\right| \cdot|\Delta x| \leqslant\left|y^{\prime}\right| \cdot \delta_{r},
$$

即 $y$ 的绝对误差限约为

$$
\delta_{y}=\left|y^{\prime}\right| \cdot \delta_{x}
$$

$y$ 的相对误差限约为

$$
\frac{\delta_{y}}{|y|}=\left|\frac{y^{\prime}}{y}\right| \cdot \delta_{x} .
$$

以后常把绝对误差限与相对误差限简称为绝对误差与相对误差.

## 习 题 $2-5$

1. 已知 $y=x^{3}-x$, 计算在 $x=2$ 处当 $\Delta x$ 分别等于 $1,0.1,0.01$ 时的 $\Delta y$ 及 $\mathrm{d} y$.
2. 设函数 $y=f(x)$ 的图形如图 2-12, 试在图 2-12(a)、(b)、(c)、(d) 中分别标出在点 $x_{0}$ 的 $\mathrm{d} y 、 \Delta y$ 及 $\Delta y-\mathrm{d} y$, 并说明其正负.

3. 求下列函数的微分:
(1) $y=\frac{1}{x}+2 \sqrt{x}$;
(2) $y=x \sin 2 x$;
(3) $y=\frac{x}{\sqrt{x^{2}+1}}$;
(4) $y=\ln ^{2}(1-x)$,
(5) $y=x^{2} \mathrm{e}^{2 . x}$;
(6) $y=\mathrm{e}^{-x} \cos (3-x)$;
(7) $y=\arcsin \sqrt{1-x^{2}}$;
(8) $y=\tan ^{2}\left(1+2 x^{2}\right)$;
(9) $y=\arctan \frac{1-x^{2}}{1+x^{2}}$;
(10) $s=A \sin (\omega t+\varphi)(A 、 \omega 、 \varphi$ 是常数 $)$.
4. 将适当的函数填入下列括号内,使等式成立:
(1) $\mathrm{d}(\quad)=2 \mathrm{~d} x$;
(2) $\mathrm{d}(\quad)=3 x \mathrm{~d} x$;
(3) $\mathrm{d}(\quad)=\cos t \mathrm{~d} t$;
(4) $\mathrm{d}(\quad)=\sin \omega x \mathrm{~d} x$;
(5) $\mathrm{d}(\cdot)=\frac{1}{1+x} \mathrm{~d} x$;
(6) $\mathrm{d}()=\mathrm{e}^{-2 . r} \mathrm{~d} x$;
(7) $\mathrm{d}()=\frac{1}{\sqrt{x}} \mathrm{~d} x$;
(8) $d(\quad)=\sec ^{2} 3 x d x$.
5. 如图 2-13 所示的电缆 $A O B$ 的长为 $s$, 跨度为 $2 l$, 电绕的报低点 $O$ 与杆顶连线 $A B$ 的 距离为 $f$, 则电缆长可按下面公式计算

$$
s=2 l\left(1+\frac{2 f^{2}}{3 l^{2}}\right) \text {, }
$$

当 $f$ 变化了 $\Delta f$ 时，电孯长的变化约为多少?

6. 设羿形的圆心角 $\alpha=60^{\circ}$, 半径 $R=100 \mathrm{~cm}$ (图 2-14). 如果 $R$ 不变, $\alpha$ 惐少 $30^{\circ}$, 问阘 形面积大约改变了多少? 又如果 $\alpha$ 不变, $R$ 增加 $1 \mathrm{~cm}$, 问扇形面积大约改变了多少?

国 $2-13$

7. 计算下列三角函数值的近似值:
(1) $\cos 29^{\circ}$;
(2) $\tan 136^{\circ}$.
8. 计算下列反三角函数值的近似值:
(1) $\arcsin 0.5002$;
(2) $\arccos 0.4995$.
9. 当 $|x|$ 较小时,证明下列近似公式:

(1) $\tan x \approx x(x$ 是角的弧度值); (2) $\ln (1+x) \approx x$;

(3) $\frac{1}{1+x} \approx 1-x$,

并计算 $\tan 45^{\prime}$ 和 $\ln 1.002$ 的近似值.

10. 计算下列各根式的近似值:
(1) $\sqrt[3]{996}$;
(2) $\sqrt[6]{65}$

-11. 计算球体体积时, 要求精确度在 $2 \%$ 以内. 问这时测然直径 $D$ 的相对误差不能超过多少?

-12. 某厂生产如图 2-15 所示的阘形板, 半径 $R=200 \mathrm{~mm}$, 要求

图 $2-15$ 中心角 $\alpha$ 为 $55^{\circ}$. 产品检验时，一般用测位弦长 / 的办法来间接测墨中心角 $\alpha$. 如果测垃弦长 / 时的误差 $\delta_{l}=0.1 \mathrm{~mm}$, 问由此而引起的中心角测量误差 $\delta_{\alpha}$ 是多少?

## 总习题二

1. 在“充分”、“必要”和“充分必要”三者中选择一个正确的填入下列空格内:

(1) $f(x)$ 在点 $x_{0}$ 可导是 $f(x)$ 在点 $x_{0}$ 连续的 条件. $f(x)$ 在点 $x_{0}$ 连续是 $f(x)$ 在点 $x_{0}$ 可导的 条件.

(2) $f(x)$ 在点 $x_{01}$ 的左导数 $f^{\prime}-\left(x_{10}\right)$ 及右导数 $f^{\prime \prime},\left(x_{n}\right)$ 都存在且相等是 $f(x)$ 在点 $x_{0}$ 可 导的 条件.

(3) $f(x)$ 在点 $x_{0}$ 可导是 $f(x)$ 在点 $x_{0}$ 可微的 条件.

2. 设 $f(x)=x(x+1)(x+2) \cdots(x+n)(n \geqslant 2)$, 则 $f^{\prime}(0)=$
3. 选择下述题中给出的四个结论中一个正确的结论:

设 $f(x)$ 在 $x=a$ 的某个邻域内有定义, 则 $f(x)$ 在 $x=a$ 处可导的一个充分条件是 ().

(A) $\lim _{h \rightarrow+\infty} h\left[f\left(a+\frac{1}{h}\right)-f(a)\right]$ 存在.

(B) $\lim _{h \rightarrow 0} \frac{f(a+2 h)-f(a+h)}{h}$ 存在.

(C) $\lim _{h \rightarrow 0} \frac{f(a+h)-f(a-h)}{2 h}$ 存在.

(D) $\lim _{h \rightarrow 0} \frac{f(a)-f(a-h)}{h}$ 存在.

4. 设有一根细棒, 取棒的一端作为原点, 棒上任意点的坐标为 $x$,于是分布在区间 $[0, x]$ 上细棒的质刋 $m$ 与 $x$ 存在函数关系 $m=m(x)$. 应怎样确定细棒在点 $x_{0}$ 处的线密度 (对于 均匀细棒来说, 单位长度细棒的质望叫做这细棒的线密度)?
5. 根据导数的定义, 求 $f(x)=\frac{1}{x}$ 的导数.
6. 求下列函数 $f(x)$ 的 $f^{\prime}-(0)$ 及 $f^{\prime}+(0)$, 又 $f^{\prime}(0)$ 是否存在:

(1) $f(x)= \begin{cases}\sin x, & x<0, \\ \ln (1+x), & x \geqslant 0 ;\end{cases}$

(2) $f(x)= \begin{cases}\frac{x}{1+\mathrm{e}^{\frac{1}{x}}}, & x \neq 0, \\ 0, & x=0 .\end{cases}$

7. 讨论函数

$$
f(x)= \begin{cases}x \sin \frac{1}{x}, & x \neq 0, \\ 0, & x=0\end{cases}
$$

在 $x=0$ 处的连续性与可导性. 8. 求下列函数的导数:
(1) $y=\arcsin (\sin x)$;
(2) $y=\arctan \frac{1+x}{1-x}$;
(3) $y=\ln \tan \frac{x}{2}-\cos x \cdot \ln \tan x$;
(4) $y=\ln \left(e^{y}+\sqrt{1+e^{2 x}}\right)$;
(5) $y=x^{\frac{1}{1}}(x>0)$.

9. 求下列函数的二阶导数:
(1) $y=\cos ^{2} x \cdot \ln x$;
(2) $y=\frac{x}{\sqrt{1-x^{2}}}$.

-10. 求下列函数的 $n$ 阶导数:
(1) $y=\sqrt[m]{1+x}$;
(2) $y=\frac{1-x}{1+x}$.

11. 设函数 $y=y(x)$ 由方程 $\mathrm{e}^{y}+x y=\mathrm{e}$ 所确定, 求 $y^{\prime \prime}(0)$.
12. 求下列由参数方程所确定的函数的一阶导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ 及二阶导数 $\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}$ :
(1) $\left\{\begin{array}{l}x=a \cos ^{3} \theta \\ y=a \sin ^{3} \theta\end{array}\right.$
(2) $\left\{\begin{array}{l}x=\ln \sqrt{1+t^{2}}, \\ y=\arctan t\end{array}\right.$
13. 求曲线 $\left\{\begin{array}{l}x=2 \mathrm{e}^{\prime} \\ y=\mathrm{e}^{-t}\end{array}\right.$ 在 $t=0$ 相应的点处的切线方程及法线方程.
14. 已知 $f(x)$ 是周期为 5 的连续函数, 它在 $x=0$ 的某个邻域内测足关系式

$$
f(1+\sin x)-3 f(1-\sin x)=8 x+o(x),
$$

且 $f(x)$ 在 $x=\mathrm{I}$ 处可导, 求曲线 $y=f(x)$ 在点 $(6, f(6))$ 处的切线方程.

15. 当正在高度 $H$ 水平飞行的飞机开始向机场跑道下降时，如图 2-16 所示从飞机到 机场的水平地面距离为 $L$. 假设飞机下降的路径为三次函数 $y=a x^{3}+b x^{2}+c x+d$ 的图形, 其中 $\left.y\right|_{, ~}=-1 .=H,\left.y\right|_{, \cdots 0}=0$. 试确定飞机的降落路径.

因 2-16

16. 甲船以 $6 \mathrm{~km} / \mathrm{h}$ 的速率向东行驶, 乙船以 $8 \mathrm{~km} / \mathrm{h}$ 的速率问南行驶. 在中午十二点整, 乙船位于甲船之北 $16 \mathrm{~km}$ 处. 问下午一点正两船相离的速宰为多少? 17. 利用函数的微分代替函数的增量求 $\sqrt[3]{1.02}$ 的近似值.
17. 已知单撰的振动周期 $T=2 \pi \sqrt{\frac{l}{g}}$, 其中 $g=980 \mathrm{~cm} / \mathrm{s}^{2}, l$ 为摆长 (单位为 $\mathrm{cm}$ ). 设原 摆长为 $20 \mathrm{~cm}$, 为使周期 $T$ 增大 $0.05 \mathrm{~s}$, 摆长约需加长多少?

## 第三章 微分中值定理与导数的应用

上一章里, 从分析实际问题中因变量相对于自变量的变化快慢出发, 引进了 导数概念, 并讨论了导数的计算方法. 本章中, 我们将应用导数来研究函数以及 曲线的某些性态,并利用这些知识解决一些实际问题. 为此, 先要介绍微分学的 几个中值定理,它们是导数应用的理论基础.

## 第一节 微分中值定理

我们先讲罗尔 (Rolle) 定理, 然后根据它推出拉格朗日 (Lagrange) 中值定理 和柯西中值定理.

## 一、罗尔定理

首先, 我们观察图 3-1. 设曲线弧 $\widehat{A B}$ 是函数 $y=f(x)(x \in[a, b])$ 的图形. 这是一条连续的曲线弧, 除端点外处处有不垂直 于 $x$ 轴的切线, 且两个端点的纵坐标相等, 即 $f(a)=f(b)$. 可以发现在曲线弧的最高点 $C$ 处或 最低点 $D$ 处,曲线有水平的切线. 如果记 $C$ 点的 横坐标为 $\xi$, 那么就有 $f^{\prime}(\xi)=0$. 现在用分析语言 把这个几何现象描述出来,就可得下面的罗尔定 理. 为了应用方便, 先介绍费马 (Fermat)引理.

费马引理 设函数 $f(x)$ 在点 $x_{0}$ 的某邻域

$$
\left.f(x) \leqslant f\left(x_{11}\right) \quad \text { (或 } f(x) \geqslant f\left(x_{11}\right)\right),
$$

那么 $f^{\prime}\left(x_{0}\right)=\mathbf{0}$.

证 不妨设 $x \in U\left(x_{10}\right)$ 时, $f(x) \leqslant f\left(x_{0}\right)$ (如果 $f(x) \geqslant f\left(x_{10}\right)$, 可以类似 地证明). 于是, 对于 $x_{0}+\Delta x \in U\left(x_{0}\right)$, 有

$$
f\left(x_{0}+\Delta x\right) \leqslant f\left(x_{0}\right),
$$

从而当 $\Delta x>0$ 时,

$$
\frac{f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)}{\Delta x} \leqslant 0 ;
$$

当 $\Delta x<0$ 时,

$$
\frac{f\left(x_{11}+\Delta x\right)-f\left(x_{11}\right)}{\Delta x} \geqslant 0 .
$$

根据函数 $f(x)$ 在 $x_{v}$ 可导的条件及极限的保号性,便得到

$$
\begin{aligned}
& f^{\prime}\left(x_{0}\right)=f^{\prime},\left(x_{10}\right)=\lim _{\Delta x \rightarrow 1^{+}} \frac{f\left(x_{11}+\Delta x\right)-f\left(x_{11}\right)}{\Delta x} \leqslant 0, \\
& f^{\prime}\left(x_{0}\right)=f^{\prime},\left(x_{11}\right)=\lim _{\Delta x \rightarrow 0^{-}} \frac{f\left(x_{0}+\Delta x\right)-f\left(x_{0}\right)}{\Delta x} \geqslant 0 .
\end{aligned}
$$

所以, $f^{\prime}\left(x_{0}\right)=0$. 证毕.

通常称导数等于零的点为函数的驻点 (或稳定点, 临界点).

罗尔定理 如果函数 $f(x)$ 满足
(1) 在闭区间 $[a, b]$ 上连续;
(2) 在开区间 $(a, b)$ 内可导;
(3) 在区间端点处的函数值相等, 即 $f(a)=f(b)$,

那么在 $(a, b)$ 内至少有一点 $\xi(a<\xi<b)$, 使得 $f^{\prime}(\xi)=0$.

证 由于 $f(x)$ 在闭区间 $[a, b]$ 上连续，根据闭区间上连续函数的最大值最 小值定理, $f(x)$ 在闭区间 $[a, b]$ 上必定取得它的最大值 $M$ 和最小值 $m$. 这样， 只有两种可能情形:

(1) $M=m$. 这时 $f(x)$ 在区间 $[a, b]$ 上必然取相同的数值 $M: f(x)=M$. 由此, $\forall x \in(a, b)$, 有 $f^{\prime}(x)=0$. 因此, 任取 $\xi \in(a, b)$, 有 $f^{\prime}(\xi)=0$.

(2) $M>m$. 因为 $f(a)=f(b)$, 所以 $M$ 和 $m$ 这两个数中至少有一个不等 于 $f(x)$ 在区间 $[a, b]$ 的端点处的函数值. 为确定起见, 不妨设 $M \neq f(a)$ (如果 设 $m \neq f(a)$, 证法完全类似), 那么必定在开区间 $(a, b)$ 内有一点 $\xi$ 使 $f(\xi)=M$. 因此, $\forall x \in[a, b]$, 有 $f(x) \leqslant f(\xi)$, 从而由费马引理可知 $f^{\prime}(\xi)=0$. 定理证 毕.

## 二、拉格朗日中值定理

罗尔定理中 $f^{\prime}(a)=f^{\prime}(b)$ 这个条件是相当特殊的, 它使罗尔定理的应用受 到限制. 如果把 $f(a)=f(b)$ 这个条件取消, 但仍保留其余两个条件, 并相应的 改变结论,那么就得到微分学中十分重要的拉格朗日中值定理.

拉格朗日中值定理 如果函数 $f(x)$ 满足

(1) 在闭区间 $[a, b]$ 上连续;

(2) 在开区间 $(a, b)$ 内可导,

那么在 $(a, b)$ 内至少有一点 $\xi(a<\xi<b)$, 使等式

$$
f(b)-f(a)=f^{\prime}(\xi)(b-a)
$$

成立.

在证明之前,先看一下定理的几何意义. 如 果把(1)式改写成

$$
\frac{f(b)-f(a)}{b-a}=f^{\prime}(\xi),
$$

由图 3-2 可看出, $\frac{f(b)-f(a)}{b-a}$ 为弦 $A B$ 的斜 率, 而 $f^{\prime}(\xi)$ 为曲线在点 $C$ 处的切线的斜率. 因 此拉格朗日中值定理的几何意义是: 如果连续 曲线 $y=f(x)$ 的弧 $\overparen{A B}$ 上除端点外处处具有不

从图 3-1 看出, 在罗尔定理中, 由于 $f(a)=f(b)$, 弦 $A B$ 是平行于 $x$ 轴 的,因此点 $C$ 处的切线实际上也平行于弦 $A B$. 由此可见,罗尔定理是拉格朗日 中值定理的特殊情形.

从上述拉格朗日中值定理与罗尔定理的关系,自然想到利用罗尔定理来证 明拉格朗日中值定理.但在拉格朗日中值定理中, 函数 $f(x)$ 不一定具备 $f(a)=$ $f(b)$ 这个条件, 为此我们设想构造一个与 $f(x)$ 有密切联系的函数 $\varphi(x)$ (称为 辅助函数), 使 $\varphi(x)$ 满足条件 $\varphi(a)=\varphi(b)$. 然后对 $\varphi(x)$ 应用罗尔定理,再把 对 $\varphi(x)$ 所得的结论转化到 $f(x)$ 上,证得所要的结果。我们从拉格朗日中值定理 的几何解释中来寻找辅助函数, 从图 3-2 中看到,有向线段 $N M$ 的值是 $x$ 的函 数, 把它表示为 $\varphi(x)$, 它与 $f(x)$ 有密切的联系, 且当 $x=a$ 及 $x=b$ 时, 点 $M$ 与点 $N$ 重合, 即有 $\varphi(a)=\varphi(b)=0$. 为求得函数 $\varphi(x)$ 的表达式, 设直线 $A B$ 的 方程为 $y=L(x)$, 则

$$
L(x)=f(a)+\frac{f(b)-f(a)}{b-a}(x-a),
$$

由于点 $M 、 N$ 的纵坐标依次为 $f(x)$ 及 $L(x)$, 故表示有向线段 $N M$ 的值的函数

$$
\varphi(x)=f(x)-L(x)=f(x)-f(a)-\frac{f(b)-f(a)}{b-a}(x-a) .
$$

下面就利用这个辅助函数来证明拉格朗日中值定理.

定理的证明引进辅助函数

$$
\varphi(x)=f(x)-f(a)-\frac{f(b)-f(a)}{b-a}(x-a) .
$$

容易验证函数 $\varphi(x)$ 适合罗尔定理的条件: $\varphi(a)=\varphi(b)=0 ; \varphi(x)$ 在闭区间 $[a, b]$ 上连续,在开区间 $(a, b)$ 内可导, 且

$$
\varphi^{\prime}(x)=f^{\prime}(x)-\frac{f(b)-f(a)}{b-a} .
$$

根据罗尔定理, 可知在 $(a, b)$ 内至少有一点 $\xi$, 使 $\varphi^{\prime}(\xi)=0$, 即

由此得

$$
f^{\prime}(\xi)-\frac{f(b)-f(a)}{b-a}=0 \text {. }
$$

即

$$
\frac{f(b)-f(a)}{b-a}=f^{\prime}(\xi) \text {, }
$$

定理证毕。

$$
f(b)-f(a)=f^{\prime}(\xi)(b-a) \text {. }
$$

显然,公式(1)对于 $b<a$ 也成立. (1)式叫做拉格朗日中值公式.

设 $x$ 为区间 $[a, b]$ 内一点, $x+\Delta x$ 为这区间内的另一点 $(\Delta x>0$ 或 $\Delta x<0$ ), 则公式 (1) 在区间 $[x, x+\Delta x]$ （当 $\Delta x>0$ 时) 或在区间 $[x+\Delta x, x]$ (当 $\Delta x<0$ 时) 上就成为

$$
f(x+\Delta x)-f(x)=f^{\prime}(x+\theta \Delta x) \cdot \Delta x \quad(0<\theta<1) .
$$

这里数值 $\theta$ 在 0 与 1 之间,所以 $x+\theta \Delta x$ 是在 $x$ 与 $x+\Delta x$ 之间.

如果记 $f(x)$ 为 $y$, 则 (2)式又可写成

$$
\Delta y=f^{\prime}(x+\theta \Delta x) \cdot \Delta x \quad(0<\theta<1) .
$$

我们知道, 函数的微分 $\mathrm{d} y=f^{\prime}(x) \cdot \Delta x$ 是函数的增㯺 $\Delta y$ 的近似表达式,一般说 来, 以 $\mathrm{d} y$ 近似代替 $\Delta y$ 时所产生的误差只有当 $\Delta x \rightarrow 0$ 吋才趋于零; 而 (3) 式却 给出了自变量取得有限增量 $\Delta x(|\Delta x|$ 不一定很小) 时, 函数增量 $\Delta y$ 的准确表 达式. 因此, 这个定理也叫做有限增量定理, (3) 式称为有限增量公式. 拉格朗日 中值定理在微分学中占有重要地位, 有时也称这定理为微分中值定理. 在某些问 题中当自变量 $x$. 取得有限增量 $\Delta x$ 而需要函数增量的准确表达式时, 拉格朗日 中值定理就显出它的价值.

作为拉格朗日中值定理的一个应用, 我们来导出以后讲积分学时很有用的 一个定理. 我们知道, 如果函数 $f(x)$ 在某一区间上是一个常数, 那么 $f(x)$ 在该 区间上的导数恒为零. 它的逆命题也是成立的, 这就是:

定理 如果函数 $f(x)$ 在区间 $I$ 上的导数恒为零, 那么 $f(x)$ 在区间 $I$ 上是 一个常数.

证在区间 $I$ 上任取两点 $x_{1}, x_{2}\left(x_{1}<x_{2}\right)$, 应用 (1)式就得

$$
f\left(x_{2}\right)-f\left(x_{1}\right)=f^{\prime}(\xi)\left(x_{2}-x_{1}\right)\left(x_{1}<\xi<x_{2}\right) .
$$

由假定, $f^{\prime}(\xi)=0$, 所以 $f\left(x_{2}\right)-f\left(x_{1}\right)=0$, 即

$$
f\left(x_{2}\right)=f\left(x_{1}\right) \text {. }
$$

因为 $x_{1} 、 x_{2}$ 是 $I$ 上任意两点, 所以上面的等式表明: $f(x)$ 在 $I$ 上的函数值总是 相等的, 这就是说, $f(x)$ 在区间 $I$ 上是一个常数. 从上述论证中可以看出, 虽然拉格朗日中值定理中的 $\xi$ 的准确数值不知 道,但在这里并不妨碍它的应用.

例 1 证明当 $x>0$ 时,

$$
\frac{x}{1+x}<\ln (1+x)<x .
$$

证 设 $f(x)=\ln (1+x)$, 显然 $f(x)$ 在区间 $[0, x]$ 上满足拉格朗日中值定 理的条件, 根据定理, 应有

$$
f(x)-f(0)=f^{\prime}(\xi)(x-0), 0<\xi<x .
$$

由于 $f(0)=0, f^{\prime}(x)=\frac{1}{1+x}$, 因此上式即为

$$
\ln (1+x)=\frac{x}{1+\xi} .
$$

又由 $0<\xi<x$, 有

$$
\frac{x}{1+x}<\frac{x}{1+\xi}<x
$$

即

$$
\frac{x}{1+x}<\ln (1+x)<x \quad(x>0) .
$$

## 三、柯西中值定理

上面已经指出, 如果连续曲线弧 $\overparen{A B}$ 上除端点外处处具有不垂直于横轴的切 线, 那么这段弧上至少有一点 $C$, 使曲线在点 $C$ 处的切线平行于弦 $A B$. 设 $\overparen{A B}$ 由 参数方程

$$
\left\{\begin{array}{l}
X=F(x), \\
Y=f(x)
\end{array} \quad(a \leqslant x \leqslant b)\right.
$$

表示(图 3-3), 其中 $x$ 为参数. 那么曲线上点 $(X, Y)$ 处的切线的斜率为

$$
\frac{\mathrm{d} Y}{\mathrm{~d} X}=\frac{f^{\prime}(x)}{F^{\prime}(x)},
$$

弦 $A B$ 的斜率为

$$
\frac{f(b)-f(a)}{F(b)-F(a)} \text {. }
$$

假定点 $C$ 对应于参数 $x=\xi$, 那么曲线上点 $C$ 处的切线平行于弦 $A B$, 可表示为

$$
\frac{f(b)-f(a)}{F(b)-\bar{F}(a)}=\frac{f^{\prime}(\xi)}{F^{\prime}(\xi)} \text {. }
$$

与这一事实相应的是

柯西中值定理，如果函数 $f(x)$ 及 $F(x)$ 满足

(1) 在闭区间 $[a, b]$ 上连续;

(2) 在开区间 $(a, b)$ 内可导;

(3) 对任一 $x \in(a, b), F^{\prime}(x) \neq 0$,

那么在 $(a, b)$ 内至少有一点 $\xi$, 使等式

$$
\frac{f(b)-f(a)}{F(b)-F(a)}=\frac{f^{\prime}(\xi)}{F^{\prime}(\xi)}
$$

成立.

证 首先注意到 $F(b)-F(a) \neq 0$. 这是由于

$$
F(b)-F(a)=F^{\prime}(\eta)(b-a),
$$

其中 $a<\eta<b$, 根据假定 $F^{\prime}(\eta) \neq 0$, 文 $b-a \neq 0$, 所以

$$
F(b)-F(a) \neq 0 \text {. }
$$

类似拉格朗日中值定理的证明, 我们仍然以表示有向线段 $N M$ 的值的函数 $\varphi(x)$ (见图 3-3) 作为辅助函数. 这里, 点 $M$ 的纵坐标为 $Y=f(x)$, 点 $N$ 的纵 坐标为

$$
Y=f(a)+\frac{f(b)-f(a)}{F(b)-F(a)}[F(x)-F(a)] \text {. }
$$

于是

$$
\varphi(x)=f(x)-f(a)-\frac{f(b)-f(a)}{F(b)-F(a)}[F(x)-F(a)] .
$$

容易验证，这个辅助函数 $\varphi(x)$ 适合罗尔定理的条件: $\varphi(a)=\varphi(b)=0$; $\varphi(x)$ 在闭区间 $[a, b]$ 上连续,在开区间 $(a, b)$ 内可导且

$$
\varphi^{\prime}(x)=f^{\prime}(x)-\frac{f(b)-f(a)}{F(b)-F(a)} \cdot F^{\prime}(x) .
$$

根据罗尔定理, 可知在 $(a, b)$ 内必定有一点 $\xi$, 使得 $\varphi^{\prime}(\xi)=0$, 即

$$
f^{\prime}(\xi)-\frac{f(b)-f(a)}{F(b)-F(a)} \cdot F^{\prime}(\xi)=0,
$$

由此得

$$
\frac{f(b)-f(a)}{F(b)-F(a)}=\frac{f^{\prime}(\xi)}{F^{\prime}(\xi)}
$$

定理证毕.

很明显, 如果取 $F(x)=x$, 那么 $F(b)-F(a)=b-a, F^{\prime}(x)=1$, 因而公 式(4)就可以写成:

$$
f^{\prime}(b)-f(a)=f^{\prime}(\xi)(b-a)(a<\xi<b),
$$

这样就变成拉格朗日中值公式了.

## 习 题 3-1

1. 验证罗尔定理对函数 $y=\ln \sin x$ 在区间 $\left[\frac{\pi}{6}, \frac{5 \pi}{6}\right]$ 上的正确性.
2. 验证拉格朗日中值定理对函数 $y=4 x^{3}-5 x^{2}+x-2$ 在区间 $[0,1]$ 上的正确性.
3. 对函数 $f(x)=\sin x$ 及 $F(x)=x+\cos x$ 在区间 $\left[0, \frac{\pi}{2}\right]$ 上验证柯西中值定理的正 确性.
4. 试证明对函数 $y=p x^{2}+q x+r$ 应用拉格朗日中值定理时所求得的点 $\xi$ 总是位于区 间的正中间.
5. 不用求出函数 $f(x)=(x-1)(x-2)(x-3)(x-4)$ 的导数, 说明方程 $f^{\prime}(x)=0$ 有 几个实根，并指出它们所在的区间.
6. 证明恒等式: $\arcsin x+\arccos x=\frac{\pi}{2}(-1 \leqslant x \leqslant 1)$.
7. 若方程 $a_{n} x^{n}+a_{1} x^{n-1}+\cdots+a_{n-1} x=0$ 有一个正根 $x=x_{0}$, 证明方程 $a_{0} n x^{n-1}+$ $a_{1}(n-1) x^{n-2}+\cdots+a_{n-1}=0$ 必有一个小于 $x_{0}$ 的正根.
8. 若函数 $f(x)$ 在 $(a, b)$ 内具有二阶导数, 且 $f\left(x_{1}\right)=f\left(x_{2}\right)=f\left(x_{3}\right)$, 其中 $a<x_{1}<x_{2}$ $<x_{3}<b$, 证明: 在 $\left(x_{1}, x_{3}\right)$ 内至少有一点 $\xi$, 使得 $f^{\prime \prime}(\xi)=0$.
9. 设 $a>b>0, n>1$, 证明:

$$
n b^{n-1}(a-b)<a^{n}-b^{n}<n a^{n-1}(a-b) .
$$

10. 设 $a>b>0$, 证明：

$$
\frac{a-b}{a}<\ln \frac{a}{b}<\frac{a-b}{b}
$$

11. 证明下列不等式:

(1) $|\arctan a-\arctan b| \leqslant|a-b|$;

(2) 当 $x>1$ 时, $\mathrm{e}^{\mathrm{r}}>\mathrm{e} \cdot x$.

12. 证明方程 $x^{5}+x-1=0$ 只有一个正根.
13. 设 $f(x) 、 g(x)$ 在 $[a, b]$ 上连续, 在 $(a, b)$ 内可导, 证明在 $(a, b)$ 内有一点 $\xi$, 使

$$
\left|\begin{array}{ll}
f(a) & f(b) \\
g(a) & g(b)
\end{array}\right|=(b-a)\left|\begin{array}{ll}
f(a) & f^{\prime}(\xi) \\
g(a) & g^{\prime}(\xi)
\end{array}\right| .
$$

14. 证明: 若函数 $f(x)$ 在 $(-\infty,+\infty)$ 内满足关系式 $f^{\prime}(x)=f(x)$, 且 $f(0)=1$, 则 $f(x)=\mathrm{e}^{x}$.
15. 设函数 $y=f(x)$ 在 $x=0$ 的某邻域内具有 $n$ 阶导数, 且 $f(0)=f(0)=\cdots$ $=f^{(n-1)}(0)=0$, 试用柯西中值定理证明:

$$
\frac{f(x)}{x^{\prime \prime}}=\frac{f^{(n)}(\theta x)}{n !}(0<\theta<1) \text {. }
$$

## 第二节 洛必达法则

如果当 $x \rightarrow a$ (或 $x \rightarrow \infty$ ) 时, 两个函数 $f(x)$ 与 $F(x)$ 都趋于零或都趋于无 穷大, 那么极限 $\lim _{\substack{x \rightarrow \infty \\(x \rightarrow \infty)}} \frac{f(x)}{F(x)}$ 可能存在、也可能不存在. 通常把这种极限叫做未定 式, 并分别简记为 $\frac{0}{0}$ 或 $\frac{\infty}{\infty}$. 在第一章第六节中讨论过的极限 $\lim _{x \rightarrow 0} \frac{\sin x}{x}$ 就是未定式 $\frac{0}{0}$ 的一个例子. 对于这类极限, 即使它存在也不能用“商的极限等于极限的商”这 一法则. 下面我们将根据柯西中值定理来推出求这类极限的一种简便且重要的 方法.

我们着重讨论 $x \rightarrow a$ 时的未定式 $\frac{0}{0}$ 的情形,关于这情形有以下定理:

定理 1 设

(1) 当 $x \rightarrow a$ 时, 函数 $f(x)$ 及 $F(x)$ 都趋于零;

(2) 在点 $a$ 的某去心邻域内, $f^{\prime}(x)$ 及 $F^{\prime}(x)$ 都存在且 $F^{\prime}(x) \neq 0$;

(3) $\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}$ 存在 (或为无穷大),

那么

$$
\lim _{x \rightarrow a} \frac{f(x)}{F(x)}=\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)} .
$$

这就是说, 当 $\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}$ 存在时, $\lim _{x \rightarrow a} \frac{f(x)}{F(x)}$ 也存在且等于 $\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}$; 当 $\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}$ 为无穷大时, $\lim _{x \rightarrow a} \frac{f(x)}{F(x)}$ 也是无穷大. 这种在一定条件下通过分子分母 分别求导再求极限来确定未定式的值的方法称为洛必达 (L'Hospital) 法赑.

证因为求 $\frac{f(x)}{F(x)}$ 当 $x \rightarrow a$ 时的极限与 $f(a)$ 及 $F(a)$ 无关, 所以可以假定 $f(a)=F(a)=0$, 于是由条件 (1)、(2)知道, $f(x)$ 及 $F(x)$ 在点 $a$ 的某一邻域内 是连续的. 设 $x$ 是这邻域内的一点, 那么在以 $x$ 及 $a$ 为端点的区间上, 柯西中值 定理的条件均满足, 因此有

令 $x \rightarrow a$, 并对上式两端求极限, 注意到 $x \rightarrow a$ 时 $\xi \rightarrow a$, 再根据条件 (3) 便得要证 明的结论.

如果 $\frac{f^{\prime}(x)}{F^{\prime}(x)}$ 当 $x \rightarrow a$ 时仍属 $\frac{0}{0}$ 型, 且这时 $f^{\prime}(x), F^{\prime}(x)$ 能满足定理中 $f(x), F(x)$ 所要满足的条件, 那么可以继续施用洛必达法则先确定 $\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}$, 从而确定 $\lim _{x \rightarrow a} \frac{f(x)}{F(x)}$, 即

$$
\lim _{x \rightarrow a} \frac{f(x)}{F(x)}=\lim _{x \rightarrow a} \frac{f^{\prime}(x)}{F^{\prime}(x)}=\lim _{x \rightarrow a} \frac{f^{\prime \prime}(x)}{F^{\prime \prime}(x)} .
$$

且可以以次类推.

例 1 求 $\lim _{x \rightarrow 01} \frac{\sin a x}{\sin b x}(b \neq 0)$.

解 $\lim _{x \rightarrow 0} \frac{\sin a x}{\sin b x}=\lim _{x \rightarrow 11} \frac{a \cos a x}{b \cos b x}=\frac{a}{b}$.

例 2 求 $\lim _{x \rightarrow 1} \frac{x^{3}-3 x+2}{x^{3}-x^{2}-x+1}$.

解 $\lim _{x \rightarrow 1} \frac{x^{3}-3 x+2}{x^{3}-x^{2}-x+1}=\lim _{x \rightarrow 1} \frac{3 x^{2}-3}{3 x^{2}-2 x-1}$

$$
=\lim _{x \rightarrow 1} \frac{6 x}{6 x-2}=\frac{3}{2} \text {. }
$$

注意, 上式中的 $\lim _{x \rightarrow 1} \frac{6 x}{6 x-2}$ 已不是未定式, 不能对它应用洛必达法则, 否则要 导致错误结果. 以后使用洛必达法则时应当经常注意这一点,如果不是未定式, 就不能应用洛必达法则。

例 3 求 $\lim _{x \rightarrow 0} \frac{x-\sin x}{x^{3}}$.

解 $\lim _{x \rightarrow 0} \frac{x-\sin x}{x^{3}}=\lim _{x \rightarrow 11} \frac{1-\cos x}{3 x^{2}}=\lim _{x \rightarrow 0} \frac{\sin x}{6 x}=\frac{1}{6}$.

我们指出,对于 $x \rightarrow \infty$ 时的未定式 $\frac{0}{0}$ 以及对于 $x \rightarrow a$ 或 $x \rightarrow \infty$ 时的未定式 $\frac{\infty}{\infty}$, 也有相应的洛必达法则. 例如, 对于 $x \rightarrow \infty$ 时的未定式 $\frac{0}{0}$ 有以下定理.

定理 2 设

(1) 当 $x \rightarrow \infty$ 时, 函数 $f(x)$ 及 $F(x)$ 都趋于雾;

(2) 当 $|x|>N$ 时 $f^{\prime}(x)$ 与 $F^{\prime}(x)$ 都存在,且 $F^{\prime}(x) \neq 0$;

(3) $\lim _{x \rightarrow \infty} \frac{f^{\prime}(x)}{F^{\prime}(x)}$ 存在 (或为无穷大),

那么

$$
\lim _{x \rightarrow \infty} \frac{f(x)}{F(x)}=\lim _{x \rightarrow \infty} \frac{f^{\prime}(x)}{F^{\prime}(x)} .
$$

例 4 求 $\lim _{x \rightarrow+\infty} \frac{\frac{\pi}{2}-\arctan x}{\frac{1}{x}}$.

解 $\lim _{x \rightarrow+\infty} \frac{\frac{\pi}{2}-\arctan x}{\frac{1}{x}}=\lim _{x \rightarrow+\infty} \frac{-\frac{1}{1+x^{2}}}{-\frac{1}{x^{2}}}=\lim _{x \rightarrow \infty} \frac{x^{2}}{1+x^{2}}=1$.

例 5 求 $\lim _{n \rightarrow \infty} \frac{\ln x}{x^{n}}(n>0)$. 解 $\lim _{x \rightarrow+\infty} \frac{\ln x}{x^{n}}=\lim _{n \rightarrow \infty} \frac{\frac{1}{x}}{n x^{n-1}}=\lim _{n \rightarrow+\infty} \frac{1}{n x^{n}}=0$.

例 6 求 $\lim _{x \rightarrow+\infty} \frac{x^{\prime \prime}}{\mathrm{e}^{\lambda x}}(n$ 为正整数, $\lambda>0)$.

解 相继应用洛必达法则 $n$ 次,得

$$
\begin{aligned}
\lim _{n \rightarrow+\infty} \frac{x^{n}}{\mathrm{e}^{i x}} & =\lim _{1 \rightarrow+\infty} \frac{n x^{n-1}}{\lambda \mathrm{e}^{\lambda x}}=\lim _{x \rightarrow+\infty} \frac{n(n-1) x^{n-2}}{\lambda^{2} \mathrm{e}^{\lambda x}}=\cdots \\
& =\lim _{x \rightarrow+\infty} \frac{n !}{\lambda^{n} \mathrm{e}^{\lambda x}}=0 .
\end{aligned}
$$

事实上,如果例 6 中的 $n$ 不是正整数而是任何正数,那么极限仍为零.

对数函数 $\ln x$ 、常函数 $x^{n}(n>0)$ 、指数函数 $\mathrm{e}^{\lambda x}(\lambda>0)$ 均为当 $x \rightarrow+\infty$ 时的 无穷大,但从例 5 、例 6 可以看出,这三个函数增大的“速度”是很不一样的,暑函 数增大的“速度”比对数函数快得多,而指数函数增大的 “速度” 又比幂函数快 得多.

下表列出了 $x=10,100,1000$ 时, 函数 $\ln x, \sqrt{x}, x^{2}$ 及 $\mathrm{e}^{x}$ 相应的函数值. 从 中可以看出当 $x$ 增大时这几个函数增大“速度”快慢的情况。

| $x$ | 10 | 100 | 1000 |
| :---: | :---: | :---: | :---: |
| $\ln x$ | 2.3 | 4.6 | 6.9 |
| $\sqrt{x}$ | 3.2 | 10 | 31.6 |
| $x^{2}$ | 100 | $10^{4}$ | $10^{6}$ |
| $\mathrm{e}^{x}$ | $2.20 \times 10^{4}$ | $2.69 \times 10^{43}$ | $1.97 \times 10^{4.34}$ |

其他还有一些 $0 \cdot \infty 、 \infty-\infty 、 0^{0} 、 1^{\infty} 、 \infty^{0}$ 型的未定式,也可通过 $\frac{0}{0}$ 或 $\frac{\infty}{\infty}$ 型的 未定式来计算,下面用例子说明.

例 7 求 $\lim _{x \rightarrow 0^{+}} x^{n} \ln x(n>0)$.

解 这是未定式 $0 \cdot \infty$. 因为

$$
x^{\prime \prime} \ln x=\frac{\ln x}{\frac{1}{x^{\prime \prime}}},
$$

当 $x \rightarrow 0^{\prime}$ 时, 上式石端是未定式 $\frac{\infty}{\infty}$, 应用洛必达法则, 得

$$
\lim _{x \rightarrow 0^{+}} x^{n} \ln x=\lim _{x \rightarrow 0^{+}} \frac{\ln x}{x^{-n}}=\lim _{x \rightarrow 0^{+}} \frac{\frac{1}{x}}{-n x^{-n-1}}=\lim _{x \rightarrow 0^{+}}\left(\frac{-x^{n}}{n}\right)=0 .
$$

例 8 求 $\lim (\sec x-\tan x)$.

$$
\lim _{x \rightarrow \frac{\pi}{2}}
$$

解 这是未定式 $\infty-\infty$. 因为

$$
\sec x-\tan x=\frac{1-\sin x}{\cos x},
$$

当 $x \rightarrow \frac{\pi}{2}$ 时, 上式右端是未定式 $\frac{0}{0}$, 应用洛必达法则, 得

$$
\lim _{x \rightarrow \frac{\pi}{2}}(\sec x-\tan x)=\lim _{x \rightarrow \frac{\pi}{2}} \frac{1-\sin x}{\cos x}=\lim _{x \rightarrow \frac{\pi}{2}} \frac{-\cos x}{-\sin x}=0 .
$$

例 9 求 $\lim _{x \rightarrow 0^{+}} x^{x}$.

解 这是未定式 $0^{0}$. 设 $y=x^{x}$, 取对数得

$$
\ln y=x \ln x \text {, }
$$

当 $x \rightarrow 0^{+}$时, 上式右端是未定式 $0 \cdot \infty$. 应用例 7 的结果,得

$$
\lim _{x \rightarrow 0^{+}} \ln y=\lim _{x \rightarrow 0^{+}}(x \ln x)=0 \text {. }
$$

因为 $y=\mathrm{e}^{\ln y}$, 而 $\lim y=\lim \mathrm{e}^{\ln y}=\mathrm{e}^{\lim \ln y}\left(\right.$ 当 $\left.x \rightarrow 0^{+}\right)$, 所以

$$
\lim _{x \rightarrow 0^{+}} x^{x}=\lim _{x \rightarrow 0^{+}} y=\mathrm{e}^{0}=1 \text {. }
$$

洛必达法则是求未定式的一种有效方法, 但最好能与其他求极限的方法结 合使用. 例如能化简时应尽可能先化简, 可以应用等价无穷小替代或重要极限 时,应尽可能应用,这样可以使运算简捷.

例 10 求 $\lim _{x \rightarrow 0} \frac{\tan x-x}{x^{2} \sin x}$.

解 如果直接用洛必达法则, 那么分母的导数(尤其是高阶导数)较繁.如果 作一个等价无穷小替代, 那么运算就方便得多. 其运算如下:

$$
\begin{aligned}
\lim _{x \rightarrow 0} \frac{\tan x-x}{x^{2} \sin x} & =\lim _{x \rightarrow 0} \frac{\tan x-x}{x^{3}} \cdot \frac{x}{\sin x}=\lim _{x \rightarrow 0} \frac{\tan x-x}{x^{3}} \\
& =\lim _{x \rightarrow 0} \frac{\sec ^{2} x-1}{3 x^{2}}=\lim _{x \rightarrow 0} \frac{2 \sec ^{2} x \tan x}{6 x}=\frac{1}{3} \lim _{x \rightarrow 0} \frac{\tan x}{x}=\frac{1}{3} .
\end{aligned}
$$

最后,我们指出,本节定理给出的是求未定式的一种方法.当定理条件满足 时, 所求的极限当然存在 (或为 $\infty$ ), 但当定理条件不满足时, 所求极限却不一定 不存在, 这就是说, 当 $\lim \frac{f^{\prime}(x)}{F^{\prime}(x)}$ 不存在时 (等于无穷大的情况除外), $\lim \frac{f(x)}{F(x)}$ 仍 可能存在 (见本节习题第 2 题).

## 习 题 $3-2$

1. 用洛必达法则求下列极限:
(1) $\lim _{x \rightarrow 0} \frac{\ln (1+x)}{x}$;
(2) $\lim _{x \rightarrow 0} \frac{\mathrm{e}^{1}-\mathrm{e}^{-x}}{\sin x}$;
(3) $\lim _{x \rightarrow a} \frac{\sin x-\sin a}{x-a}$;
(4) $\lim _{x \rightarrow \infty} \frac{\sin 3 x}{\tan 5 x}$;
(5) $\lim _{x \rightarrow \frac{\pi}{2}} \frac{\ln \sin x}{(\pi-2 x)^{2}}$;
(6) $\lim _{x \rightarrow a} \frac{x^{\prime \prime \prime}-a^{m}}{x^{n}-a^{n}}(a \neq 0)$;
(7) $\lim _{x \rightarrow 0^{+}} \frac{\ln \tan 7 x}{\ln \tan 2 x}$;
(8) $\lim _{x \rightarrow \frac{\pi}{2}} \frac{\tan x}{\operatorname{lan} 3 x}$;
(9) $\lim _{, \rightarrow+\infty} \frac{\ln \left(1+\frac{1}{x}\right)}{\operatorname{arccot} x}$;
(10) $\lim _{x \rightarrow 11} \frac{\ln \left(1+x^{2}\right)}{\sec x-\cos x}$;
(11) $\lim _{x \rightarrow 0} x \cot 2 x$;
(12) $\lim _{x \rightarrow 0} x^{2} \mathrm{e}^{16 x^{2}}$;
(13) $\lim _{x \rightarrow 1}\left(\frac{2}{x^{2}-1}-\frac{1}{x-1}\right)$;
(14) $\lim _{x \rightarrow \infty}\left(1+\frac{a}{x}\right)^{x}$;
(15) $\lim _{x \rightarrow 11^{+}} x^{\mathrm{Nin} x}$;
(16) $\lim _{x \rightarrow 0^{+}}\left(\frac{1}{x}\right)^{\tan x}$;
2. 验证极限 $\lim _{x \rightarrow \infty} \frac{x+\sin x}{x}$ 存在,但不能用洛必达法则得出.
3. 验证极限 $\lim _{x \rightarrow 0} \frac{x^{2} \sin \frac{1}{x}}{\sin x}$ 存在,但不能用洛必达法则得出.

4. 讨论函数

$$
f(x)=\left\{\begin{array}{cc}
{\left[\frac{(1+x)^{\frac{1}{x}}}{\mathrm{e}}\right]^{\frac{1}{x}},} & x>0 \\
\mathrm{e}^{-\frac{1}{2}}, & x \leqslant 0
\end{array}\right.
$$

在点 $x=0$ 处的连续性.

## 第三节 泰 勒 公式

对于一些较复杂的函数,为了便于研究,往往希望用一些简单的函数来近似 表达. 由于用多项式表示的函数, 只要对自变量进行有限次加、减、乘三种算术运 算,便能求出它的函数值来,因此我们经常用多项式来近似表达函数.

在微分的应用中已经知道, 当 $|x|$ 很小时,有如下的近似等式:

$$
\mathrm{e}^{x} \approx 1+x, \ln (1+x) \approx x .
$$

这些都是用一次多项式来近似表达函数的例子. 显然, 在 $x=0$ 处这些一次多项 式及其一阶导数的值,分别等于被近似表达的函数及其导数的相应值.

但是这种近似表达式还存在着不足之处:首先是精确度不高, 它所产生的误 差仅是关于 $x$ 的高阶无穷小; 其次是用它来作近似计算时, 不能具体估算出误 差大小. 因此, 对于精确度要求较高且需要估计误差的时候, 就必须用高次多项 式来近似表达函数,同时给出误差公式.

于是提出如下的问题: 设函数 $f(x)$ 在含有 $x_{11}$ 的开区间内具有直到 $(n+1)$ 阶导数,试找出一个关于 $\left(x-x_{n}\right)$ 的 $n$ 次多项式

$$
p_{n}(x)=a_{11}+a_{1}\left(x-x_{0}\right)+a_{2}\left(x-x_{0}\right)^{2}+\cdots+a_{n}\left(x-x_{11}\right)^{n}
$$

来近似表达 $f^{\prime}(x)$, 要求 $p_{n}(x)$ 与 $f(x)$ 之差是比 $\left(x-x_{n}\right)$ 高阶的无穷小, 并给 出误差 $\left|f(x)-p_{n}(x)\right|$ 的具体表达式.

下面我们来讨论这个问题. 假设 $p_{n}(x)$ 在 $x_{0}$ 处的函数值及它的直到 $n$ 阶 导数在 $x_{01}$ 处的值依次与 $f\left(x_{0}\right), f^{\prime}\left(x_{0}\right), \cdots, f^{(n)}\left(x_{11}\right)$ 相等, 即满足

$$
\begin{aligned}
& p_{n}\left(x_{0}\right)=f\left(x_{0}\right), p^{\prime}{ }_{n}\left(x_{0}\right)=f^{\prime}\left(x_{0}\right), \\
& p^{\prime \prime}\left(x_{0}\right)=f^{\prime \prime}\left(x_{0}\right), \cdots, p_{n}^{(n)}\left(x_{0}\right)=f^{(n)}\left(x_{11}\right),
\end{aligned}
$$

按这些等式来确定多项式 (1) 的系数 $a_{11}, a_{1}, a_{2}, \cdots, a_{n}$. 为此, 对 (1) 式求各阶导 数,然后分别代入以上等式, 得

$$
\begin{gathered}
a_{0}=f\left(x_{0}\right), \quad 1 \cdot a_{1}=f^{\prime}\left(x_{01}\right), \\
2 ! a_{2}=f^{\prime \prime}\left(x_{01}\right), \cdots, \quad n ! a_{n}=f^{(n)}\left(x_{0}\right),
\end{gathered}
$$

即得

$$
a_{0}=f\left(x_{0}\right), a_{1}=f^{\prime}\left(x_{11}\right), a_{2}=\frac{1}{2 !} f^{\prime \prime}\left(x_{0}\right), \cdots, a_{n}=\frac{1}{n !} f^{(n)}\left(x_{0}\right) .
$$

将求得的系数 $a_{0}, a_{1}, a_{2}, \cdots, a_{n}$ 代入(1)式,有

$$
\begin{gathered}
p_{n}(x)=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{11}\right)}{2 !}\left(x-x_{11}\right)^{2}+\cdots+ \\
\frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{01}\right)^{\prime \prime} .
\end{gathered}
$$

下面的定理表明，多项式(2)的确是所要找的 $n$ 次多项式.

泰勒 (Taylor) 中值定理 如果函数 $f(x)$ 在含有 $x_{0}$ 的某个开区间 $(a, b)$ 内 具有直到 $(n+1)$ 阶的导数, 则对任一 $x \in(a, b)$, 有

$$
\begin{gathered}
f(x)=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !}\left(x-x_{0}\right)^{2}+\cdots+ \\
\frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right) n+R_{n}(x),
\end{gathered}
$$

其中

$$
R_{n}(x)=\frac{f^{(n+1)}(\xi)}{(n+1) !}\left(x-x_{n}\right)^{n+1},
$$

这里 $\xi$ 是 $x_{0}$ 与 $x$ 之间的某个值.

$$
\left.R_{n}(x)=\frac{f^{(n+1)}(\xi)}{(n+1) !}\left(x-x_{0}\right)^{n+1} \quad \text { ( } \xi \text { 在 } x_{0} \text { 与 } x \text { 之间 }\right) .
$$

由假设可知, $R_{n}(x)$ 在 $(a, b)$ 内具有直到 $(n+1)$ 阶的导数, 且

$$
R_{n}\left(x_{0}\right)=R_{n}^{\prime}\left(x_{0}\right)=R_{n}^{\prime \prime}\left(x_{0}\right)=\cdots=R_{n}^{(n)}\left(x_{0}\right)=0 .
$$

对两个函数 $R_{n}(x)$ 及 $\left(x-x_{0}\right)^{n+1}$ 在以 $x_{0}$ 及 $x$ 为端点的区间上应用柯西中值 定理(显然,这两个函数满足柯西中值定理的条件), 得

$$
\frac{R_{n}(x)}{\left(x-x_{0}\right)^{n+1}}=\frac{R_{n}(x)-R_{n}\left(x_{0}\right)}{\left(x-x_{0}\right)^{n \prime \prime}-0}=\frac{R_{n}^{\prime}\left(\xi_{1}\right)}{(n+1)\left(\xi_{1}-x_{0}\right)^{\prime \prime}}
$$

( $\xi_{1}$ 在 $x_{11}$ 与 $x$ 之间),

再对两个函数 $R_{n}^{\prime}(x)$ 与 $(n+1)\left(x-x_{0}\right)$ "在以 $x_{0}$ 及 $\xi_{1}$ 为端点的区间上应用柯 西中值定理, 得

$$
\begin{aligned}
\frac{R_{n}^{\prime}\left(\xi_{1}\right)}{(n+1)\left(\xi_{1}-x_{0}\right)^{n}} & =\frac{R_{n}^{\prime}\left(\xi_{1}\right)-R_{n}^{\prime}\left(x_{0}\right)}{(n+1)\left(\xi_{1}-x_{0}\right)^{n}-0} \\
& =\frac{R_{n}^{\prime \prime}\left(\xi_{2}\right)}{n(n+1)\left(\xi_{2}-x_{0}\right)^{n-1}}
\end{aligned}
$$

( $\xi_{2}$ 在 $x_{0}$ 与 $\xi_{1}$ 之间).

照此方法继续做下去, 经过 $(n+1)$ 次后，得

$$
\frac{R_{n}(x)}{\left(x-x_{0}\right)^{n+1}}=\frac{R_{n}^{(n+1)}(\xi)}{(n+1) !}
$$

( $\xi$ 在 $x_{0}$ 与 $\xi_{n}$ 之间, 因面也在 $x_{0}$ 与 $x$ 之间).

注意到 $R_{n}^{(n+1)}(x)=f^{(n+1)}(x)$ (因 $p_{n}^{(n+1)}(x)=0$ ), 则由上式得

$$
R_{n}(x)=\frac{f^{(n+1)}(\xi)}{(n+1) !}\left(x-x_{0}\right)^{n+1} \quad \text { ( } \xi \text { 在 } x_{0} \text { 与 } x \text { 之间), }
$$

定理证毕.

多项式 (2) 称为函数 $f(x)$ 按 $\left(x-x_{0}\right)$ 的幂展开的 $n$ 次泰勒多项式,公式 (3) 称为 $f(x)$ 按 $\left(x-x_{0}\right)$ 的并展开的带有拉格朗日型余项的 $n$ 阶泰勒公式,而 $R_{n}(x)$ 的表达式 (4)称为拉格朗日型余项.

当 $n=0$ 时,泰勒公式变成拉格朗日中值公式

因此, 泰勒中值定理是拉格朗日中值定理的推广.

由泰勒中值定理可知, 以多项式 $p_{n}(x)$ 近似表达函数 $f(x)$ 时, 其误差为 $\left|R_{n}(x)\right|$. 如果对于某个固定的 $n$, 当 $x \in(a, b)$ 时, $\left|f^{(n+1)}(x)\right| \leqslant M$, 则有估计 式

$$
\left|R_{n}(x)\right|=\left|\frac{f^{(n+1)}(\xi)}{(n+1) !}\left(x-x_{0}\right)^{n+1}\right| \leqslant \frac{M}{(n+1) !}\left|x-x_{0}\right|^{n+1}
$$

及

$$
\lim _{x \rightarrow x_{0}} \frac{R_{n}(x)}{\left(x-x_{0}\right)^{n}}=0 .
$$

由此可见, 当 $x \rightarrow x_{\mathrm{n}}$ 时误差 $\left|R_{n}(x)\right|$ 是比 $\left(x-x_{0}\right)^{n}$ 高阶的无穷小, 即

$$
R_{n}(x)=o\left[\left(x-x_{0}\right)^{n}\right] \text {. }
$$

这样,我们提出的问题圆满地得到解决.

在不需要余项的精确表达式时, $n$ 阶泰勒公式也可写成

$$
f(x)=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\cdots+\frac{f^{(\prime \prime)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right)^{n}+o\left[\left(x-x_{0}\right)^{n}\right] .
$$

$R_{n}(x)$ 的表达式 (6)称为佩亚诺(Peano)型余项, 公式 (7) 称为 $f(x)$ 按 $(x-$ $\left.x_{0}\right)$ 的幂展开的带有佩亚诺型余项的 $n$ 阶泰勒公式(1).

在泰勒公式 (3)中, 如果取 $x_{0}=0$, 则 $\xi$ 在 0 与 $x$ 之间. 因此可以令 $\xi=\theta x$ $(0<\theta<1)$, 从而泰勒公式变成较简单的形式, 即所谓带有拉格朗日型余项的麦 克劳林 (Maclaurin) 公式

$$
\begin{gathered}
f(x)=f(0)+f^{\prime}(0) x+\frac{f^{\prime \prime}(0)}{2 !} x^{2}+\cdots+\frac{f^{(n)}(0)}{n !} x^{n}+ \\
\frac{f^{(n+1)}(\theta x)}{(n+1) !} x^{n+1} \quad(0<\theta<1) .
\end{gathered}
$$

在泰勒公式(7)中,如果取 $x_{0}=0$, 则有带有佩亚诺型余项的麦克劳林公式

$$
f(x)=f(0)+f^{\prime}(0) x+\cdots+\frac{f^{(n)}(0)}{n !} x^{n}+o\left(x^{n}\right) .
$$

由(8)或(9)可得近似公式

$$
f(x) \approx f(0)+f^{\prime}(0) x+\frac{f^{\prime \prime}(0)}{2 !} x^{2}+\cdots+\frac{f^{(\prime \prime)}(0)}{n !} x^{n},
$$

误差估计式(5)相应的变成

$$
\left|R_{n}(x)\right| \leqslant \frac{M}{(n+1) !}|x|^{n+1} .
$$

例 1 写出函数 $f(x)=\mathrm{e}^{r}$ 的带有拉格朗日型余项的 $n$ 阶麦克劳林公式.

解 因为

$$
f^{\prime}(x)=f^{\prime \prime}(x)=\cdots=f^{(n)}(x)=\mathrm{e}^{x} \text {, }
$$

所以

$$
f(0)=f^{\prime}(0)=f^{\prime \prime}(0)=\cdots=f^{(n)}(0)=1 \text {. }
$$

把这些值代入公式 $(8)$,并注意到 $f^{(n+1)}(\theta x)=\mathrm{e}^{\theta_{x}}$ 便得

(1) 这里公式 (7) 是 $f^{(n+1)}(x)$ 在区间 $(a, b)$ 内有界的条件下推得的. 事实上公式(7)只要在“ $f(x)$ 在 含有 $x_{0}$ 的开区间 $(a, b)$ 内具有直到 $n$ 阶的导数，且 $f^{(n)}(x)$ 在 $(a, b)$ 内连续”们条件下就成立.

$$
\mathrm{e}^{x}=1+x+\frac{x^{2}}{2 !}+\cdots+\frac{x^{n}}{n !}+\frac{\mathrm{e}^{a_{r}}}{(n+1) !} x^{\prime \prime \prime} \quad(0<\theta<1) .
$$

由这个公式可知, 若把 $\mathrm{e}^{x}$ 用它的 $n$ 次泰勒多项式表达为

$$
\mathrm{e}^{x} \approx 1+x+\frac{x^{2}}{2 !}+\cdots+\frac{x^{n}}{n !},
$$

这时所产生的误差为

$$
\left|R_{n}(x)\right|=\left|\frac{\mathrm{e}^{\theta x}}{(n+1) !} x^{n+1}\right|<\frac{\mathrm{e}^{1 . r i}}{(n+1) !}|x|^{n+1} \quad(0<\theta<1) .
$$

如果取 $x=1$, 则得无理数 $\mathrm{e}$ 的近似式为

$$
\mathrm{e} \approx 1+1+\frac{1}{2 !}+\cdots+\frac{1}{n !},
$$

其误差

$$
\left|R_{n}\right|<\frac{\mathrm{e}}{(n+1) !}<\frac{3}{(n+1) !} .
$$

当 $n=10$ 时, 可算出 $\mathrm{e} \approx 2.718282$, 其误差不超过 $10^{-6}$.

例 2 求 $f(x)=\sin x$ 的带有拉格朗日型余项的 $n$ 阶麦克劳林公式.

解 因为

$$
\begin{gathered}
f^{\prime}(x)=\cos x, f^{\prime \prime}(x)=-\sin x, f^{\prime \prime}(x)=-\cos x, \\
f^{(4)}(x)=\sin x, \cdots, f^{(n)}(x)=\sin \left(x+\frac{n \pi}{2}\right),
\end{gathered}
$$

所以

$$
f(0)=0, f^{\prime}(0)=1, f^{\prime \prime}(0)=0, f^{\prime \prime \prime}(0)=-1, f^{(4)}(0)=0
$$

等等. 它们顺序循环地取四个数 $0,1,0,-1$, 于是按公式(8)得 (令 $n=2 m$ )

$$
\sin x=x-\frac{x^{3}}{3 !}+\frac{x^{5}}{5 !}-\cdots+(-1)^{m-1} \frac{x^{2 m-1}}{(2 m-1) !}+R_{2 m},
$$

其中

$$
R_{2 m}(x)=\frac{\sin \left[\theta x+(2 m+1) \frac{\pi}{2}\right]}{(2 m+1) !} x^{2 m+1} \quad(0<\theta<1) .
$$

如果取 $m=1$, 则得近似公式

$$
\sin x \approx x,
$$

这时误差为

$$
\left|R_{2}\right|=\left|\frac{\sin \left(\theta x+\frac{3}{2} \pi\right)}{3 !} x^{3}\right| \leqslant \frac{|x|^{3}}{6} \quad(0<\theta<1) .
$$

如果 $m$ 分别取 2 和 3 , 则可得 $\sin x$ 的 3 次和 5 次泰勒多项式

$$
\sin x \approx x-\frac{1}{3 !} x^{3} \text { 和 } \sin x \approx x-\frac{1}{3 !} x^{3}+\frac{1}{5 !} x^{5},
$$

其误差的绝对值依次不超过 $\frac{1}{5 !}|x|^{5}$ 和 $\frac{1}{7 !}|x|^{7}$. 以上三个泰勒多项式及正弦函 数的图形都画在图 3-4 中, 以便于比较.

类似地,还可以得到

$$
\cos x=1-\frac{1}{2 !} x^{2}+\frac{1}{4 !} x^{4}-\cdots+(-1)^{m} \frac{1}{(2 m) !} x^{2 m}+R_{2 m+1}(x),
$$

其中 $R_{2 m+1}(x)=\frac{\cos [\theta x+(m+1) \pi]}{(2 m+2) !} x^{2 m+2}(0<\theta<1)$;

$$
\ln (1+x)=x-\frac{1}{2} x^{2}+\frac{1}{3} x^{3}-\cdots+(-1)^{n-1} \frac{1}{n} x^{n}+R_{n}(x),
$$

其中 $R_{n}(x)=\frac{(-1)^{n}}{(n+1)(1+\theta x)^{n+1}} x^{n+1} \quad(0<\theta<1)$;

$$
(1+x)^{\alpha}=1+\alpha x+\frac{\alpha(\alpha-1)}{2 !} x^{2}+\cdots+\frac{\alpha(\alpha-1) \cdots(\alpha-n+1)}{n !} x^{n}+R_{u}(x),
$$

其中 $R_{n}(x)=\frac{\alpha(\alpha-1) \cdots(\alpha-n+1)(\alpha-n)}{(n+1) !}(1+\theta x)^{a-n-1} x^{n+1}(0<\theta<1)$.

由以上带有拉格朗日型余项的麦克劳林公式, 易得相应的带有佩亚诺型余 项的麦克劳林公式,读者可自行写出.

例 3 利用带有佩亚诺型余项的麦克劳林公式, 求极限 $\lim _{x \rightarrow 0} \frac{\sin x-x \cos x}{\sin ^{3} x}$.

解 由于分式的分母 $\sin ^{3} x \sim x^{3}(x \rightarrow 0)$, 我们只需将分子中的 $\sin x$ 和 $x \cos x$ 分别用带有佩亚诺型余项的三阶麦克劳林公式表示, 即

$$
\begin{aligned}
& \sin x=x-\frac{x^{3}}{3 !}+o\left(x^{3}\right), \\
& x \cos x=x-\frac{x^{3}}{2 !}+o\left(x^{3}\right) .
\end{aligned}
$$

于是

$$
\begin{aligned}
\sin x-x \cos x & =x-\frac{x^{3}}{3 !}+o\left(x^{3}\right)-x+\frac{x^{3}}{2 !}-o\left(x^{3}\right) \\
& =\frac{1}{3} x^{3}+o\left(x^{3}\right)
\end{aligned}
$$

对上式作运算时, 把两个比 $x^{3}$ 高阶的无穷小的代数和仍记作 $o\left(x^{3}\right)$, 故

$$
\lim _{x \rightarrow 0} \frac{\sin x-x \cos x}{\sin ^{3} x}=\lim _{x \rightarrow 0} \frac{\frac{1}{3} x^{3}+o\left(x^{3}\right)}{x^{3}}=\frac{1}{3} .
$$

## 习 题 3-3

1. 按 $(x-4)$ 的算展开多项式 $f(x)=x^{4}-5 x^{3}+x^{2}-3 x+4$.
2. 应用麦克劳林公式, 按 $x$ 的箱展开函数 $f(x)=\left(x^{2}-3 x+1\right)^{3}$.
3. 求函数 $f(x)=\sqrt{x}$ 按 $(x-4)$ 的程展开的带有拉格朗日型余项的 3 阶泰勒公式.
4. 求函数 $f(x)=\ln x$ 按 $(x-2)$ 的年展开的带有浘亚诺型余项的 $n$ 阶泰勒公式.
5. 求函数 $f(x)=\frac{1}{x}$ 按 $(x+1)$ 的箱展开的带有拉格朗日型余项的 $n$ 阶泰勒公式.
6. 求函数 $f(x)=\tan x$ 的带有佩亚诺型余项的 3 阶麦克劳林公式.
7. 求函数 $f(x)=x \mathrm{e}^{r}$ 的带有栭亚诺型余项的 $n$ 阶麦克劳林公式.
8. 验证当 $0<x \leqslant \frac{1}{2}$ 时, 按公式 $\mathrm{e}^{x} \approx 1+x+\frac{x^{2}}{2}+\frac{x^{3}}{6}$ 计算 $\mathrm{e}^{x}$ 的近似值时, 所产生的误差 小于 0.01 ,并求 $\sqrt{\mathrm{e}}$ 的近似值,使误差小于 0.01 .
9. 应用 3 阶泰勒公式求下列各数的近似值，并估计误差：
(1) $\sqrt[3]{30}$;
(2) $\sin 18^{\circ}$.

10. 利用泰勒公式求下列极限:

(1) $\lim _{x \rightarrow+\infty}\left(\sqrt[3]{x^{3}+3 x^{2}}-\sqrt[4]{x^{4}-2 x^{3}}\right)$;

(2) $\lim _{x \rightarrow 0} \frac{\cos x-\mathrm{e}^{-\frac{x^{2}}{2}}}{x^{2}[x+\ln (1-x)]}$;

(3) $\lim _{x \rightarrow 0} \frac{1+\frac{1}{2} x^{2}-\sqrt{1+x^{2}}}{\left(\cos x-\mathrm{e}^{x^{2}}\right) \sin x^{2}}$.

第四节 函数的单调性与曲线的凹凸性

## 一、函数单调性的判定法

第一章第一节中已经介绍了函数在区间上单调的概念.下面利用导数来对 函数的单调性进行研究.

如果函数 $y=f(x)$ 在 $[a, b]$ 上单调增加 (单调减少), 那么它的图形是一条 沿 $x$ 轴正向上升 (下降) 的曲线. 这时, 如图 3-5, 曲线上各点处的切线斜率是非 负的 (是非正的), 即 $y^{\prime}=f^{\prime}(x) \geqslant 0\left(y^{\prime}=f^{\prime}(x) \leqslant 0\right)$. 由此可见, 函数的单调性 与导数的符号有着密切的联系.

反过来,能否用导数的符号来判定函数的单调性呢?

下面我们利用拉格朗日中值定理来进行讨论.

设函数 $f(x)$ 在 $[a, b]$ 上连续,在 $(a, b)$ 内可导，在 $[a, b]$ 上任取两点 $x_{1} 、 x_{2}$ $\left(x_{1}<x_{2}\right)$, 应用拉格朗日中值定理,得到

$$
f\left(x_{2}\right)-f\left(x_{1}\right)=f^{\prime}(\xi)\left(x_{2}-x_{1}\right)\left(x_{1}<\xi<x_{2}\right) .
$$

由于在(1)式中, $x_{2}-x_{1}>0$, 因此, 如果在 $(a, b)$ 内导数 $f^{\prime}(x)$ 保持正号, 即 $f^{\prime}(x)>0$, 那么也有 $f^{\prime}(\xi)>0$. 于是

$$
f\left(x_{2}\right)-f\left(x_{1}\right)=f^{\prime}(\xi)\left(x_{2}-x_{1}\right)>0,
$$

即

$$
f\left(x_{1}\right)<f\left(x_{2}\right),
$$

表明函数 $y=f(x)$ 在 $[a, b]$ 上单调增加. 同理,如果在 $(a, b)$ 内导数 $f^{\prime}(x)$ 保持 负号, 即 $f^{\prime}(x)<0$, 那么 $f^{\prime}(\xi)<0$, 于是 $f\left(x_{2}\right)-f\left(x_{1}\right)<0$, 即 $f\left(x_{1}\right)>f\left(x_{2}\right)$, 表明函数 $y=f(x)$ 在 $[a, b]$ 上单调减少.

归纳以上讨论, 即得

定理 1 设函数 $y=f(x)$ 在 $[a, b]$ 上连续,在 $(a, b)$ 内可导.

(1) 如果在 $(a, b)$ 内 $f^{\prime}(x)>0$, 那么函数 $y=f(x)$ 在 $[a, b]$ 上单调增加 ;

(2) 如果在 $(a, b)$ 内 $f^{\prime}(x)<0$, 那么函数 $y=f(x)$ 在 $[a, b]$ 上单调减少.

如果把这个判定法中的闭区间换成其他各种区间（包括无穷区间）,那么结 论也成立.

例 1 判定函数 $y=x-\sin x$ 在 $[0,2 \pi]$ 上的单调性.

解 因为在 $(0,2 \pi)$ 内

$$
y^{\prime}=1-\cos x>0,
$$

所以由定理 1 可知, 函数 $y=x-\sin x$ 在 $[0,2 \pi]$ 上单调增加.

例 2 讨论函数 $y=\mathrm{e}^{x}-x-1$ 的单调性.

解 $y^{\prime}=\mathrm{e}^{x}-1$.

函数 $y=\mathrm{e}^{x}-x-1$ 的定义域为 $(-\infty,+\infty)$. 因为在 $(-\infty, 0)$ 内 $y^{\prime}<0$, 所 以函数 $y=\mathrm{e}^{x}-x-1$ 在 $(-\infty, 0]$ 上单调减少; 因为在 $(0,+\infty)$ 内 $y^{\prime}>0$, 所以函 数 $y=\mathrm{e}^{x}-x-1$ 在 $[0,+\infty)$ 上单调增加.

例 3 讨论函数 $y=\sqrt[3]{x^{2}}$ 的单调性.

解 这函数的定义域为 $(-\infty,+\infty)$.

当 $x \neq 0$ 时,这函数的导数为

$$
y^{\prime}=\frac{2}{3 \sqrt[3]{x}}
$$

当 $x=0$ 时, 函数的导数不存在. 在 $(-\infty, 0)$ 内, $y^{\prime}<0$, 因此函数 $y=\sqrt[3]{x^{2}}$ 在 $(-\infty, 0]$ 上单调减少. 在 $(0,+\infty)$ 内, $y^{\prime}>0$, 因此函数 $y=\sqrt[3]{x^{2}}$ 在 $[0,+\infty)$ 上单 调增加. 函数的图形如图 3-6 所示.

我们注意到, 在例 2 中, $x=0$ 是函数 $y=$ $\mathrm{e}^{x}-x-1$ 的单调减少区间 $(-\infty, 0]$ 与单调增加 区间 $[0,+\infty)$ 的分界点, 而在该点处 $y^{\prime}=0$. 在 例 3 中, $x=0$ 是函数 $y=\sqrt[3]{x^{2}}$ 的单调诚少区 间 $(-\infty, 0]$ 与单调增加区间 $[0,+\infty)$ 的分界 点,而在该点处导数不存在.

从例 2 中看出,有些函数在它的定义区间

上不是单调的, 但是当我们用导数等于零的点来划分函数的定义区间以后, 就可 以使函数在各个部分区间上单调. 这个结论对于在定义区间上具有连续导数的 函数都是成立的. 从例 3 中可看出, 如果函数在某些点处不可导, 则划分函数的 定义区间的分点, 还应包括这些导数不存在的点. 综合上述两种情形, 我们有如 下结论:

如果函数在定义区间上连续, 除去有限个导数不存在的点外导数存在且连 续, 那么只要用方程 $f^{\prime}(x)=0$ 的根及 $f^{\prime}(x)$ 不存在的点来划分函数 $f(x)$ 的定 义区间, 就能保证 $f^{\prime}(x)$ 在各个部分区间内保持固定符号, 因而函数 $f(x)$ 在每 个部分区间上单调.

例 4 确定函数 $f(x)=2 x^{3}-9 x^{2}+12 x-3$ 的单调区间.

解 “这函数的定义域为 $(-\infty,+\infty)$. 求这函数的导数

$$
f^{\prime}(x)=6 x^{2}-18 x+12=6(x-1)(x-2) \text {. }
$$

解方程 $f^{\prime}(x)=0$, 即解

$$
6(x-1)(x-2)=0,
$$

得出它在函数定义域 $(-\infty,+\infty)$ 内的两个根 $x_{1}=1 、 x_{2}=2$. 这两个根把 $(-\infty,+\infty)$ 分成三个部分区间 $(-\infty, 1]$ 、[ 1,2$]$ 及 $[2,+\infty)$.

在区间 $(-\infty, 1)$ 内, $x-1<0 、 x-2<0$, 所以 $f^{\prime}(x)>0$. 因此, 函数 $f(x)$ 在 $(-\infty, 1]$ 内单调增加. 在区间 $(1,2)$ 内, $x-1>0 、 x-2<0$, 所以 $f^{\prime}(x)<0$. 因 此, 函数 $f(x)$ 在 $[1,2]$ 上单调减少. 在区间 $(2,+\infty)$ 内, $x-1>0 、 x-2>0$, 所以 $f^{\prime}(x)>0$. 因此, 函数 $f(x)$ 在 $[2,+\infty)$ 上单调增加.

函数 $y=f(x)$ 的图形如图 3-7 所示.

例 5 讨论函数 $y=x^{3}$ 的单调性.

解 这函数的定义域为 $(-\infty,+\infty)$.

函数的导数 $y^{\prime}=3 x^{2}$. 显然, 除了点 $x=0$ 使 $y^{\prime}=0$ 外, 在其余各点处均有 $y^{\prime}>0$. 因此函数 $y=x^{3}$ 在区间 $(-\infty, 0]$ 及 $[0,+\infty)$ 上都是单调增加的, 从而在 整个定义域 $(-\infty,+\infty)$ 内是单调增加的. 在 $x=0$ 处曲线有一水平切线. 函数 的图形如图 3-8 所示.

一般的, 如果 $f^{\prime}(x)$ 在某区间内的有限个点处为零, 在其余各点处均为正 (或负) 时,那么 $f(x)$ 在该区间上仍旧是单调增加(或单调减少)的.

下面我们举一个利用函数的单调性证明不等式的例子.

例 6 证明: 当 $x>1$ 时, $2 \sqrt{x}>3-\frac{1}{x}$.

证 令 $f(x)=2 \sqrt{x}-\left(3-\frac{1}{x}\right)$, 则

$$
f^{\prime}(x)=\frac{1}{\sqrt{x}}-\frac{1}{x^{2}}=\frac{1}{x^{2}}(x \sqrt{x}-1) .
$$

$f(x)$ 在 $[1,+\infty)$ 上连续, 在 $(1,+\infty)$ 内 $f^{\prime}(x)>0$, 因此在 $[1,+\infty)$ 上 $f(x)$ 单调增加, 从而当 $x>1$ 时, $f(x)>f(1)$.

由于 $f(1)=0$, 故 $f(x)>f^{\prime}(1)=0$, 即

$$
2 \sqrt{x}-\left(3-\frac{1}{x}\right)>0
$$

亦即

$$
2 \sqrt{x}>3-\frac{1}{x}(x>1)
$$

## 二、曲线的凹凸性与拐点

在第一目中,我们研究了函数单调性的判定法. 函数的单调性反映在图形 上, 就是曲线的上升或下降. 但是, 曲线在上升或下降的过程中, 还有一个弯曲方 向的问题.例如,图 3-9 中有两条曲线弧, 虽然它们 都是上升的,但图形却有显著的不同, $\overparen{A C B}$ 是向上凸 的曲线弧, 而 $\overparen{A D B}$ 是向上凹的曲线弧，它们的凹凸 性不同,下面我们就来研究曲线的凹凸性及其判 定法.

我们从几何上看到,在有的曲线弧上,如果任取 两点,则联结这两点间的弦总位于这两点间的弧段 的上方(图 3-10(a)), 而有的曲线弧, 则正好相反 (图3-10(b)). 曲线的这种性质就是曲线的凹凸性.

定义 设 $f(x)$ 在区间 $I$ 上连续, 如果对 $I$ 上任意两点 $x_{1}, x_{2}$ 恒有

$$
f\left(\frac{x_{1}+x_{2}}{2}\right)<\frac{f\left(x_{1}\right)+f\left(x_{2}\right)}{2},
$$

那么称 $f(x)$ 在 $I$ 上的图形是 (向上) 凹的 (或凹弧); 如果恒有

$$
f\left(\frac{x_{1}+x_{2}}{2}\right)>\frac{f\left(x_{1}\right)+f\left(x_{2}\right)}{2},
$$

那么称 $f(x)$ 在 $I$ 上的图形是(向上)凸的(或凸弧).

如果函数 $f(x)$ 在 $I$ 内具有二阶导数, 那么可以利用二阶导数的符号来判 定曲线的凹凸性, 这就是下面的曲线凹凸性的判定定理. 我们仅就 $I$ 为闭区间 的情形来叙述定理,当 $I$ 不是闭区间时,定理类同.

定理 2 设 $f(x)$ 在 $[a, b]$ 上连续, 在 $(a, b)$ 内具有一阶和二阶导数, 那么

(1) 若在 $(a, b)$ 内 $f^{\prime \prime}(x)>0$, 则 $f(x)$ 在 $[a, b]$ 上的图形是凹的;

(2) 若在 $(a, b)$ 内 $f^{\prime \prime}(x)<0$, 则 $f(x)$ 在 $[a, b]$ 上的图形是凸的.

证明 在情形 (1), 设 $x_{1}$ 和 $x_{2}$ 为 $[a, b]$ 内任意两点,且 $x_{1}<x_{2}$, 记 $\frac{x_{1}+x_{2}}{2}$ $=x_{0}$, 并记 $x_{2}-x_{0}=x_{0}-x_{1}=h$, 则 $x_{1}=x_{0}-h, x_{2}=x_{0}+h$, 由拉格朗日中值

公式,得

$$
\begin{aligned}
& f\left(x_{0}+h\right)-f\left(x_{0}\right)=f^{\prime}\left(x_{0}+\theta_{1} h\right) h, \\
& f\left(x_{0}\right)-f\left(x_{0}-h\right)=f^{\prime}\left(x_{0}-\theta_{2} h\right) h,
\end{aligned}
$$

其中 $0<\theta_{1}<1,0<\theta_{2}<1$. 两式相减, 即得

$$
f\left(x_{0}+h\right)+f\left(x_{0}-h\right)-2 f\left(x_{0}\right)=\left[f^{\prime}\left(x_{0}+\theta_{1} h\right)-f^{\prime}\left(x_{0}-\theta_{2} h\right)\right] h .
$$

对 $f^{\prime}(x)$ 在区间 $\left[x_{0}-\theta_{2} h, x_{0}+\theta_{1} h\right]$ 上再利用拉格朗日中值公式,得

$$
\left[f^{\prime}\left(x_{0}+\theta_{1} h\right)-f^{\prime}\left(x_{0}-\theta_{2} h\right)\right] h=f^{\prime \prime}(\xi)\left(\theta_{1}+\theta_{2}\right) h^{2} \text {, }
$$

其中 $x_{0}-\theta_{2} h<\xi<x_{0}+\theta_{1} h$. 按情形 (1) 的假设, $f^{\prime \prime}(\xi)>0$, 故有

$$
\begin{gathered}
f\left(x_{0}+h\right)+f\left(x_{0}-h\right)-2 f\left(x_{0}\right)>0, \\
\frac{f\left(x_{0}+h\right)+f\left(x_{0}-h\right)}{2}>f\left(x_{0}\right), \\
\frac{f\left(x_{1}\right)+f\left(x_{2}\right)}{2}>f\left(\frac{x_{1}+x_{2}}{2}\right),
\end{gathered}
$$

所以 $f(x)$ 在 $[a, b]$ 上的图形是凹的.

类似地可证明情形(2).

例 7 判定曲线 $y=\ln x$ 的凹凸性.

解 因为 $y^{\prime}=\frac{1}{x}, y^{\prime \prime}=-\frac{1}{x^{2}}$, 所以在函数 $y=\ln x$ 的定义域 $(0,+\infty)$ 内, $y^{\prime \prime}<0$, 由定理 2 可知, 曲线 $y=\ln x$ 是凸的.

例 8 判定曲线 $y=x^{3}$ 的凹凸性.

解 因为 $y^{\prime}=3 x^{2}, y^{\prime \prime}=6 x$. 当 $x<0$ 时, $y^{\prime \prime}<0$, 所以曲线在 $(-\infty, 0]$ 内为 凸弧; 当 $x>0$ 时, $y^{\prime \prime}>0$, 所以曲线在 $[0,+\infty$ )内为凹弧(参看图 3-8). 一般的, 设 $y=f(x)$ 在区间 $I$ 上连续, $x_{10}$ 是 $I$ 的内点 ${ }^{(1)}$. 如果曲线 $y=f(x)$ 在经过点 $\left(x_{0}, f\left(x_{0}\right)\right)$ 时, 曲线的凹凸性改变了, 那么就称点 $\left(x_{0}, f\left(x_{0}\right)\right)$ 为这曲 线的拐点.

如何来寻找曲线 $y=f(x)$ 的拐点呢?

从上面的定理知道, 由 $f^{\prime \prime}(x)$ 的符号可以判定曲线的凹凸性, 因此, 如果 $f^{\prime \prime}(x)$ 在 $x_{0}$ 的左、右两侧邻近异号, 那么点 $\left(x_{10}, f\left(x_{0}\right)\right)$ 就是曲线的一个拐点, 所以, 要寻找拐点, 只要找出 $f^{\prime \prime}(x)$ 符号发生变化的分界点即可. 如果 $f(x)$ 在区 间 $(a, b)$ 内具有二阶连续导数, 那么在这样的分界点处必然有 $f^{\prime \prime}(x)=0$; 除此 以外, $f(x)$ 的二阶导数不存在的点,也有可能是 $f^{\prime \prime}(x)$ 的符号发生变化的分界 点. 综合以上分析, 我们可以按下列步骤来判定区间 $I$ 上的连续曲线 $y=f(x)$ 的 拐点:

(1) 求 $f^{\prime \prime}(x)$;

(2) 令 $f^{\prime \prime}(x)=0$, 解出这方程在区间 $I$ 内的实根, 并求出在区间 $I$ 内 $f^{\prime \prime}(x)$ 不存在的点;

(3) 对于 (2) 中求出的每一个实根或二阶导数不存在的点 $x_{0}$, 检查 $f^{\prime \prime}(x)$ 在 $x_{0}$ 左、右两侧邻近的符号, 那么当两侧的符号相反时, 点 $\left(x_{0}, f\left(x_{0}\right)\right)$ 是拐点, 当 两侧的符号相同时,点 $\left(x_{0}, f\left(x_{0}\right)\right)$ 不是拐点.

例 9 求曲线 $y=2 x^{3}+3 x^{2}-12 x+14$ 的拐点.

$$
\text { 解 } y^{\prime}=6 x^{2}+6 x-12, y^{\prime \prime}=12 x+6=12\left(x+\frac{1}{2}\right) \text {. }
$$

解方程 $y^{\prime \prime}=0$, 得 $x=-\frac{1}{2}$. 当 $x<-\frac{1}{2}$ 时, $y^{\prime \prime}<0$; 当 $x>-\frac{1}{2}$ 时, $y^{\prime \prime}>0$. 因此, 点 $\left(-\frac{1}{2}, 20 \frac{1}{2}\right)$ 是这曲线的拐点.

例 10 求曲线 $y=3 x^{4}-4 x^{3}+1$ 的拐点及凹、凸的区间.

解 函数 $y=3 x^{4}-4 x^{3}+1$ 的定义域为 $(-\infty,+\infty)$.

$$
\begin{gathered}
y^{\prime}=12 x^{3}-12 x^{2}, \\
y^{\prime \prime}=36 x^{2}-24 x=36 x\left(x-\frac{2}{3}\right) .
\end{gathered}
$$

解方程 $y^{\prime \prime}=0$, 得 $x_{1}=0, x_{2}=\frac{2}{3}$.

$x_{1}=0$ 及 $x_{2}=\frac{2}{3}$ 把函数的定义域 $(-\infty,+\infty)$ 分成三个部分区间:

(1) 区间 I 的内点是指除端点外的 $I$ 内的点. $(-\infty, 0],\left[0, \frac{2}{3}\right] \cdot\left[\frac{2}{3},+\infty\right)$

在 $(-\infty, 0)$ 内, $y^{\prime \prime}>0$, 因此在区间 $(-\infty, 0]$ 上这曲线是凹的. 在 $\left(0, \frac{2}{3}\right)$ 内, $y^{\prime \prime}<0$, 因此在区间 $\left[0, \frac{2}{3}\right]$ 上这曲线是凸的. 在 $\left(\frac{2}{3},+\infty\right)$ 内, $y^{\prime \prime}>0$, 因此在区间 $\left[\frac{2}{3},+\infty\right)$ 上这曲线是凹的.

当 $x=0$ 时, $y=1$, 点 $(0,1)$ 是这曲线的一个拐点. 当 $x=\frac{2}{3}$ 时 $y=\frac{11}{27}$, 点 $\left(\frac{2}{3}, \frac{11}{27}\right)$ 也是这曲线的拐点.

例 11 问曲线 $y=x^{4}$ 是否有拐点?

解 $y^{\prime}=4 x^{3}, y^{\prime \prime}=12 x^{2}$.

显然, 只有 $x=0$ 是方程 $y^{\prime \prime}=0$ 的根. 但当 $x \neq 0$ 时, 无论 $x<0$ 或 $x>0$ 都 有 $y^{\prime \prime}>0$, 因此点 $(0,0)$ 不是这曲线的拐点. 曲线 $y=x^{4}$ 没有拐点, 它在 $(-\infty$, $+\infty)$ 内是凹的.

例 12 求曲线 $y=\sqrt[3]{x}$ 的拐点.

解 这函数在 $(-\infty,+\infty)$ 内连续, 当 $x \neq 0$ 时,

$$
y^{\prime}=\frac{1}{3 \sqrt[3]{x^{2}}}, y^{\prime \prime}=-\frac{2}{9 x \sqrt[3]{x^{2}}},
$$

当 $x=0$ 时, $y^{\prime}, y^{\prime \prime}$ 都不存在. 故二阶导数在 $(-\infty,+\infty)$ 内不连续且不具有零 点. 但 $x=0$ 是 $y^{\prime \prime}$ 不存在的点, 它把 $(-\infty,+\infty)$ 分成两个部分区间: $(-\infty, 0]$ 、 $[0,+\infty)$.

在 $(-\infty, 0)$ 内, $y^{\prime \prime}>0$, 这曲线在 $(-\infty, 0]$ 上是凹的. 在 $(0,+\infty)$ 内, $y^{\prime \prime}<0$, 这曲线在 $[0,+\infty)$ 上是凸的.

当 $x=0$ 时, $y=0$, 点 $(0,0)$ 是这曲线的一个拐点.

## 习 题 3-4

1. 判定函数 $f(x)=\arctan x-x$ 的单调性.
2. 判定函数 $f(x)=x+\cos x(0 \leqslant x \leqslant 2 \pi)$ 的单调性.
3. 确定下列函数的单调区间:
(1) $y=2 x^{3}-6 x^{2}-18 x-7$;
(2) $y=2 x+\frac{8}{x} \quad(x>0)$;
(3) $y=\frac{10}{4 x^{3}-9 x^{2}+6 . x}$;
(4) $y=\ln \left(x+\sqrt{1+x^{2}}\right)$; (5) $y=(x-1)(x+1)^{3}$;

(6) $y=\sqrt[3]{(2 x-a)(a-x)^{2}} \quad(a>0)$;

(7) $y=x^{n} \mathrm{e}^{-. r} \quad(n>0, x \geqslant 0)$;

(8) $y=x+|\sin 2 x|$.

4. 设函数 $f(x)$ 在定义域内可导, $y=f(x)$ 的图形如图 311 所示, 则导函数 $f^{\prime}(x)$ 的图形为图 3-12 中所示的四个图形 中的哪一个?
5. 证明下列不等式:

（1）当 $x>0$ 时, $1+\frac{1}{2} x>\sqrt{1+x}$;

（2）当 $x>0$ 时, $1+x \ln \left(x+\sqrt{1+x^{2}}\right)>\sqrt{1+x^{2}}$;

（3）当 $0<x<\frac{\pi}{2}$ 时, $\sin x+\tan x>2 x$;

（4）当 $0<x<\frac{\pi}{2}$ 时, $\tan x>x+\frac{1}{3} x^{3}$;

（5）当 $x>4$ 时, $2^{r}>x^{2}$.

1. 讨论方程 $\ln x=a x$ (其中 $a>0$ ) 有几个实根?
2. 单调函数的导函数是否必为单调函数? 研究下面这个例子:

$$
f(x)=x+\sin x \text {. }
$$

8. 判定下列曲线的凹凸性:
(1) $y=4 x-x^{2}$;
(2) $y=\operatorname{sh} x$;
(3) $y=x+\frac{1}{x} \quad(x>0)$;
(4) $y=x \arctan x$.
9. 求下列函数图形的拐点及凹或凸的区间:
(1) $y=x^{3}-5 x^{2}+3 x+5$;
(2) $y=x \mathrm{e}^{-. x}$;
(3) $y=(x+1)^{4}+\mathrm{e}^{-x}$;
(4) $y=\ln \left(x^{2}+1\right)$;
(5) $y=\mathrm{e}^{\text {artent } x}$;
(6) $y=x^{4}(12 \ln x-7)$.
10. 利用函数图形的凹凸性,证明下列不等式:

(1) $\frac{1}{2}\left(x^{\prime \prime}+y^{\prime \prime}\right)>\left(\frac{x+y}{2}\right)^{\prime \prime} \quad(x>0, y>0, x \neq y, n>1)$; (2) $\frac{\mathrm{e}^{r}+\mathrm{e}^{y}}{2}>\mathrm{e}^{\frac{x+y}{2}}(x \neq y)$;

(3) $x \ln x+y \ln y>(x+y) \ln \frac{x+y}{2} \quad(x>0, y>0, x \neq y)$.

11. 试证明曲线 $y=\frac{x-1}{x^{2}+1}$ 有三个拐点位于同一直线上.

12. 问 $a, b$ 为何值时, 点 $(1,3)$ 为曲线 $y=a x^{3}+b x^{2}$ 的拐点?
13. 试决定曲线 $y=a x^{3}+b x^{2}+c x+d$ 中的 $a 、 b 、 c 、 d$, 使得 $x=-2$ 处曲线有水平切线, $(1,-10)$ 为拐点, 且点 $(-2,44)$ 在曲线上.
14. 试块定 $y=k\left(x^{2}-3\right)^{2}$ 中 $k$ 的值,使曲线的拐点处的法线通过原点.

15. 设 $y=f(x)$ 在 $x=x_{0}$ 的某邻域内具有三阶连续导数, 如果 $f^{\prime}\left(x_{0}\right)=0$, 而 $f^{\prime \prime}\left(x_{0}\right) \neq 0$, 试问 $\left(x_{0}, f\left(x_{0}\right)\right)$ 是否为拐点? 为什么?

## 第五节 函数的极值与最大值最小值

## 一、函数的极值及其求法

在上节例 4 中我们看到, 点 $x=1$ 及 $x=2$ 是函数

$$
f(x)=2 x^{3}-9 x^{2}+12 x-3
$$

的单调区间的分界点. 例如, 在点 $x=1$ 的左侧邻近, 函数 $f(x)$ 是单调增加的, 在点 $x=1$ 的右侧邻近, 函数 $f(x)$ 是单调诚少的. 因此, 存在点 $x=1$ 的一个去 心邻域, 对于这去心邻域内的任何点 $x, f(x)<f(1)$ 均成立. 类似地, 关于点 $x$ $=2$, 也存在着一个去心邻域, 对于这去心邻域内的任何点 $x, f(x)>f(2)$ 均成 立 (参看图 3-7). 具有这种性质的点如 $x=1$ 及 $x=2$, 在应用上有着重要的意 义,值得我们对此作一般性的讨论.

定义 设函数 $f(x)$ 在点 $x_{0}$ 的某邻域 $U\left(x_{0}\right)$ 内有定义, 如果对于去心邻域 $\stackrel{\circ}{U}\left(x_{0}\right)$ 内的任一 $x$, 有

$$
f(x)<f\left(x_{0}\right) \quad\left(\text { 或 } f(x)>f\left(x_{0}\right)\right),
$$

那么就称 $f\left(x_{0}\right)$ 是函数 $f(x)$ 的一个极大值 (或极小值).

函数的极大值与极小值统称为函数的极值, 使函数取得极值的点称为极值 点. 例如, 上节例 4 中的函数

$$
f(x)=2 x^{3}-9 x^{2}+12 x-3
$$

有极大值 $f(1)=2$ 和极小值 $f(2)=1$, 点 $x=1$ 和 $x=2$ 是函数 $f(x)$ 的极值点.

函数的极大值和极小值概念是局部性的. 如果 $f\left(x_{0}\right)$ 是函数 $f(x)$ 的一个极 大值, 那只是就 $x_{11}$ 附近的一个局部范围来说, $f\left(x_{0}\right)$ 是 $f(x)$ 的一个最大值; 如 果就 $f(x)$ 的整个定义域来说, $f\left(x_{0}\right)$ 不见得是最大值. 关于极小值也类似.

在图 3-13 中, 函数 $f(x)$ 有两个极大值: $f\left(x_{2}\right) 、 f\left(x_{5}\right)$, 三个极小值: $f\left(x_{1}\right) 、 f\left(x_{4}\right) 、 f\left(x_{6}\right)$, 其中极大值 $f\left(x_{2}\right)$ 比极小值 $f\left(x_{6}\right)$ 还小. 就整个区间 $[a, b]$ 来说, 只有一个极小值 $f\left(x_{1}\right)$ 同时也是最小值, 而没有一个极大值是最大值.

从图中还可看到, 在函数取得极值处, 曲线的切线是水平的. 但曲线上有水 平切线的地方, 函数不一定取得极值. 例如图中 $x=x_{3}$ 处, 曲线上有水平切线, 但 $f\left(x_{3}\right)$ 不是极值.

由本章第一节费马引理可知, 如果函数 $f(x)$ 在 $x_{0}$ 处可导, 且 $f(x)$ 在 $x_{11}$ 处取得极值, 那么 $f^{\prime}\left(x_{0}\right)=0$. 这就是可导函数取得极值的必要条件. 现将此结 论叙述成如下定理.

定理 1 (必要条件) 设函数 $f(x)$ 在 $x_{0}$ 处可导, 且在 $x_{0}$ 处取得极值, 那么 $f^{\prime}\left(x_{0}\right)=0$.

定理 1 就是说: 可导函数 $f(x)$ 的极值点必定是它的驻点. 但反过来, 函数的 驻点却不一定是极值点. 例如, $f(x)=x^{3}$ 的导数 $f^{\prime}(x)=3 x^{2}, f^{\prime}(0)=0$, 因此 $x=0$ 是这可导函数的驻点, 但 $x=0$ 却不是这函数的极值点. 所以, 函数的驻点 只是可能的极值点. 此外, 函数在它的导数不存在的点处也可能取得极值. 例如, 函数 $f(x)=|x|$ 在点 $x=0$ 处不可导, 但函数在该点取得极小值.

怎样判定函数在驻点或不可导的点处究竟是否取得极值? 如果是的话, 究 竟取得极大值还是极小值? 下面给出两个判定极值的充分条件.

定理 2(第一充分条件) 设函数 $f(x)$ 在 $x_{0}$ 处连续,且在 $x_{0}$ 的某去心邻域 $\stackrel{U}{U}\left(x_{0}, \delta\right)$ 内可导.

(1) 若 $x \in\left(x_{0}-\delta, x_{0}\right)$ 时, $f^{\prime}(x)>0$, 而 $x \in\left(x_{0}, x_{0}+\delta\right)$ 时, $f^{\prime}(x)<0$, 则 $f(x)$ 在 $x_{0}$ 处取得极大值;

(2) 若 $x \in\left(x_{0}-\delta, x_{0}\right)$ 时, $f^{\prime}(x)<0$, 而 $x \in\left(x_{0}, x_{0}+\delta\right)$ 时, $f^{\prime}(x)>0$, 则 $f(x)$ 在 $x_{0}$ 处取得极小值; (3) 若 $x \in U^{\circ}\left(x_{0}, \delta\right)$ 时, $f^{\prime}(x)$ 的符号保持不变, 则 $f(x)$ 在 $x_{0}$ 处没有 极值.

证 事实上, 就情形 (1) 来说, 根据函数单调性的判定法, 函数 $f(x)$ 在 $\left(x_{10}-\delta, x_{11}\right)$ 内单调增加, 而在 $\left(x_{10}, x_{11}+\delta\right)$ 内单调减少, 又由于函数 $f(x)$ 在 $x_{0}$ 处是连续的, 故当 $x \in \dot{U}\left(x_{0}, \delta\right)$ 时, 总有 $f(x)<f\left(x_{0}\right)$. 所以, $f\left(x_{0}\right)$ 是 $f(x)$ 的 一个极大值(图 3-14(a)).

类似地可论证情形 (2)(图 3-14(b)) 及情形(3)(图 3-14(c)、(d)).

定理 2 也可简单地这样说: 当 $x$ 在 $x_{0}$ 的邻近渐增地经过 $x_{0}$ 时, 如果 $f^{\prime}(x)$ 的符号由正变负,那么 $f(x)$ 在 $x_{0}$ 处取得极大值; 如果 $f^{\prime}(x)$ 的符号由负变正, 那么 $f(x)$ 在 $x_{0}$ 处取得极小值; 如果 $f^{\prime}(x)$ 的符号并不改变, 那么 $f(x)$ 在 $x_{11}$ 处没有极值.

根据上面的两个定理,如果函数 $f(x)$ 在所讨论的区间内连续, 除个别点外处 处可导,那么就可以按下列步骤来求 $f(x)$ 在该区间内的极值点和相应的极值:

(1) 求出导数 $f^{\prime}(x)$;

(2) 求出 $f(x)$ 的全部驻点与不可导点;

(3) 考察 $f^{\prime}(x)$ 的符号在每个驻点或不可导点的左、右邻近的情形，以确定 该点是否为极值点; 如果是极值点, 进一步确定是极大值点还是极小值点; (4) 求出各极值点的函数值, 就得函数 $f^{\prime}(x)$ 的全部极值.

例 1 求函数 $f(x)=(x-4) \sqrt[3]{(x+1)^{2}}$ 的极值.

解 (1) $f(x)$ 在 $(-\infty,+\infty)$ 内连续, 除 $x=-1$ 外处处可导, 且

$$
f^{\prime}(x)=\frac{5(x-1)}{3 \sqrt[3]{x+1}}
$$

(2) 令 $f^{\prime}(x)=0$, 得驻点 $x=1, x=-1$ 为 $f(x)$ 的不可导点;

(3) 在 $(-\infty,-1)$ 内, $f^{\prime}(x)>0$; 在 $(-1,1)$ 内, $f^{\prime}(x)<0$. 故不可导点 $x=$ -1 是一个极大值点; 又在 $(1,+\infty)$ 内, $f^{\prime}(x)>0$, 故驻点 $x=1$ 是一个极小 值点;

(4) 极大值为 $f(-1)=0$, 极小值为 $f(1)=-3 \sqrt[3]{4}$.

当函数 $f(x)$ 在驻点处的二阶导数存在且不为零时, 也可以利用下述定理 来判定 $f(x)$ 在驻点处取得极大值还是极小值.

定理 3(第二充分条件) 设函数 $f(x)$ 在 $x_{0}$ 处具有二阶导数且 $f^{\prime}\left(x_{10}\right)=0$, $f^{\prime \prime}\left(x_{0}\right) \neq 0$, 那么

(1) 当 $f^{\prime \prime}\left(x_{0}\right)<0$ 时, 函数 $f(x)$ 在 $x_{0}$ 处取得极大值;

(2) 当 $f^{\prime \prime}\left(x_{0}\right)>0$ 时, 函数 $f(x)$ 在 $x_{0}$ 处取得极小值.

证 在情形 (1), 由于 $f^{\prime \prime}\left(x_{n}\right)<0$, 按二阶导数的定义有

$$
f^{\prime \prime}\left(x_{11}\right)=\lim _{x \rightarrow x_{0}} \frac{f^{\prime}(x)-f^{\prime}\left(x_{0}\right)}{x-x_{0}}<0 .
$$

根据函数极限的局部保号性, 当 $x$ 在 $x_{0}$ 的足够小的去心邻域内时,

$$
\frac{f^{\prime}(x)-f^{\prime}\left(x_{0}\right)}{x-x_{0}}<0 \text {. }
$$

但 $f^{\prime}\left(x_{0}\right)=0$, 所以上式即

$$
\frac{f^{\prime}(x)}{x-x_{11}}<0 .
$$

从而知道, 对于这去心邻域内的 $x$ 来说, $f^{\prime}(x)$ 与 $x-x_{0}$ 符号相反. 因此, 当 $x-x_{0}<0$ 即 $x<x_{0}$ 时, $f^{\prime}(x)>0$; 当 $x-x_{0}>0$ 即 $x>x_{0}$ 时, $f^{\prime}(x)<0$. 于是 根据定理 2 知道, $f(x)$ 在点 $x_{0}$ 处取得极大值.

类似地可以证明情形 $(2)$.

定理 3 表明, 如果函数 $f(x)$ 在驻点 $x_{0}$ 处的二阶导数 $f^{\prime \prime}\left(x_{11}\right) \neq 0$, 那么该驻 点 $x_{0}$ 一定是极值点, 并且可以按二阶导数 $f^{\prime \prime}\left(x_{10}\right)$ 的符号来判定 $f\left(x_{0}\right)$ 是极大 值还是极小值. 但如果 $f^{\prime \prime}\left(x_{0}\right)=0$, 定理 3 就不能应用. 事实上, 当 $f^{\prime}\left(x_{10}\right)=0$, $f^{\prime \prime}\left(x_{0}\right)=0$ 时, $f(x)$ 在 $x_{11}$ 处可能有极大值, 也可能有极小值, 也可能没有极值. 例如, $f_{1}(x)=-x^{4}, f_{2}(x)=x^{4}, f_{3}(x)=x^{3}$ 这三个函数在 $x=0$ 处就分别属于 这三种情况. 因此, 如果函数在驻点处的二阶导数为零, 那么还得用一阶导数在 驻点左右邻近的符号来判定.

例 2 求函数 $f(x)=\left(x^{2}-1\right)^{3}+1$ 的极值.

解 $f^{\prime}(x)=6 x\left(x^{2}-1\right)^{2}$.

令 $f^{\prime}(x)=0$, 求得驻点 $x_{1}=-1, x_{2}=0, x_{3}=1$.

$f^{\prime \prime}(x)=6\left(x^{2}-1\right)\left(5 x^{2}-1\right)$.

因 $f^{\prime \prime}(0)=6>0$, 故 $f(x)$ 在 $x=0$ 处取得极小值， 极小值为 $f(0)=0$.

因 $f^{\prime \prime}(-1)=f^{\prime \prime}(1)=0$, 故用定理 3 无法判别. 考 察一阶导数 $f^{\prime}(x)$ 在驻点 $x_{1}=-1$ 及 $x_{3}=1$ 左右邻近 的符号:

当 $x$ 取 -1 左侧邻近的值时, $f^{\prime \prime}(x)<0$; 当 $x$ 取 -1 右侧邻近的值时, $f^{\prime}(x)<0$; 因为 $f^{\prime}(x)$ 的符号没 有改变, 所以 $f(x)$ 在 $x=-1$ 处没有极值. 同理,

## 二、最大值最小值问题

在工农业生产、工程技术及科学实验中, 常常会遇到这样一类问题: 在一定 条件下，怎样使“产品最多”、“用料最省”、“成本最低”、“效率最高”等问题,这类 问题在数学上有时可归结为求某一函数 (通常称为目标函数) 的最大值或最小值 问题.

假定函数 $f(x)$ 在闭区间 $[a, b]$ 上连续, 在开区间 $(a, b)$ 内除有限个点外可 导, 且至多有有限个驻点. 在上述条件下, 我们来讨论 $f(x)$ 在 $[a, b]$ 上的最大值 和最小值的求法.

首先, 由闭区间上连续函数的性质, 可知 $f(x)$ 在 $[a, b]$ 上的最大值和最小 值一定存在.

其次, 如果最大值 (或最小值) $f\left(x_{0}\right)$ 在开区间 $(a, b)$ 内的点 $x_{11}$ 处取得, 那 么,按 $f(x)$ 在开区间内除有限个点外可导且至多有有限个驻点的假定, 可知 $f\left(x_{0}\right)$ 一定也是 $f(x)$ 的极大值 (或极小值), 从而 $x_{0}$ 一定是 $f(x)$ 的驻点或不可 导点. 又 $f(x)$ 的最大值和最小值也可能在区间的端点处取得. 因此, 可用如下 方法求 $f(x)$ 在 $[a, b]$ 上的最大值和最小值.

(1) 求出 $f(x)$ 在 $(a, b)$ 内的驻点 $x_{1}, x_{2}, \cdots, x_{m}$ 及不可导点 $x^{\prime}, x_{2}^{\prime}, \cdots$, $x^{\prime}$; (2) 计算 $f\left(x_{i}\right)(i=1,2, \cdots, m), f\left(x_{j}^{\prime}\right)(j=1,2, \cdots, n)$ 及 $f(a), f(b)$;

(3) 比较 (2) 中诸值的大小, 其中最大的便是 $f(x)$ 在 $[a, b]$ 上的最大值, 最 小的便是 $f(x)$ 在 $[a, b]$ 上的最小值.

例 3 求函数 $f(x)=\left|x^{2}-3 x+2\right|$ 在 $[-3,4]$ 上的最大值与最小值.

$$
\text { 解 } \begin{aligned}
f(x) & = \begin{cases}x^{2}-3 x+2, & x \in[-3,1] \cup[2,4], \\
-x^{2}+3 x-2, & x \in(1,2) .\end{cases} \\
f^{\prime}(x) & = \begin{cases}2 x-3, & x \in(-3,1) \cup(2,4), \\
-2 x+3, & x \in(1,2) .\end{cases}
\end{aligned}
$$

在 $(-3,4)$ 内, $f(x)$ 的驻点为 $x=\frac{3}{2}$; 不可导点为 $x=1,2$.

由于 $f(-3)=20, f(1)=0, f\left(\frac{3}{2}\right)=\frac{1}{4}, f(2)=0, f(4)=6$, 比较可得 $f(x)$ 在 $x=-3$ 处取得它在 $[-3,4]$ 上的最大值 20 , 在 $x=1$ 和 $x=2$ 处取得它在 $[-3,4]$ 上的最小值 0 .

例 4 铁路线上 $A B$ 段的距离为 $100 \mathrm{~km}$. 工厂 $C$ 距 $A$ 处为 $20 \mathrm{~km}, A C$ 垂直 于 $A B$ (图 3-16). 为了运输需要, 要在 $A B$ 线 上选定一点 $D$ 向工厂修筑一条公路. 已知铁 路每公里货运的运费与公路上每公里货运的 运费之比为 3:5. 为了使货物从供应站 $B$ 运到 工厂 $C$ 的运费最省,问 $D$ 点应选在何处?

解 设 $A D=x \mathrm{~km}$, 那么 $D B=100-x$,

$$
C D=\sqrt{20^{2}+x^{2}}=\sqrt{400+x^{2}} .
$$

由于铁路上每公里货运的运费与公路上每公里货运的运费之比为 3:5, 因 此我们不妨设铁路上每公里的运费为 $3 k$, 公路上每公里的运费为 $5 k$ ( $k$ 为某个 正数, 因它与本题的解无关, 所以不必定出). 设从 $B$ 点到 $C$ 点需要的总运费为 $y$, 那么

$$
\begin{gathered}
y=5 k \cdot C D+3 k \cdot D B, \\
\text { 即 } y=5 k \sqrt{400+x^{2}}+3 k(100-x) \quad(0 \leqslant x \leqslant 100) .
\end{gathered}
$$

现在,问题就归结为: $x$ 在 $[0,100]$ 内取何值时目标函数 $y$ 的值最小. 先求 $y$ 对 $x$ 的导数:

$$
y^{\prime}=k\left(\frac{5 x}{\sqrt{400+x^{2}}}-3\right)
$$

解方程 $y^{\prime}=0$, 得 $x=15 \mathrm{~km}$.

由于 $\left.y\right|_{x=0}=400 k,\left.y\right|_{x=15}=380 k,\left.y\right|_{x=100}=500 k \sqrt{1+\frac{1}{5^{2}}}$, 其中以 $\left.y\right|_{. x=15}$ $=380 k$ 为最小, 因此, 当 $A D=x=15 \mathrm{~km}$ 时, 总运虹为最省.

在求函数的最大值 (或最小值) 时, 特别值得指出的是下述情形: $f(x)$ 在一 个区间 (有限或无限, 开或闭)内可导且只有一个驻点 $x_{0}$, 并且这个驻点 $x_{0}$ 是函 数 $f(x)$ 的极值点, 那么, 当 $f\left(x_{0}\right)$ 是极大值时, $f\left(x_{11}\right)$ 就是 $f(x)$ 在该区间上的 最大值 (图 3-17(a)) ; 当 $f\left(x_{11}\right)$ 是极小值时, $f\left(x_{11}\right)$ 就是 $f(x)$ 在该区间上的最 小值(图 3-17(b)). 在应用问题中往往遇到这样的情形.

国 $3-17$

例 5 一束光线由空气中 $A$ 点经过水面折射后到达水中 $B$ 点 (图 3-18). 已知光在空气中和水中传播的速度分别是 $v_{1}$ 和 $v_{2}$, 光线在介质中总是沿着耗 时最少的路径传播. 试确定光线传播的路径.

解. 设 $A$ 点到水面的垂直距离为 $A O=h_{1}, B$ 点到水面的垂直距离为 $B Q$ $=h_{2}, x$ 轴沿水面过点 $O 、 Q, O Q$ 的长度为 $l$.

由于光线总是沿着耗时最少的路径传播, 因 此光线在同一均匀介质中必沿直线传播. 设光线 的传播路径与 $x$ 轴的交点为 $P, O P=x$, 则光线从 $A$ 到 $B$ 的传播路径必为折线 $A P B$, 其所需要的传 播时间为

$$
T(x)=\frac{\sqrt{h_{1}^{2}+x^{2}}}{v_{1}}+\frac{\sqrt{h_{2}^{2}+(l-x)^{2}}}{v_{2}}, x \in[0, l] .
$$

下面来确定 $x$ 满足什么条件时, $T(x)$ 在 $[0, \iota]$ 上取得最小值.

国 $3-18$

由于

$$
\begin{gathered}
T^{\prime}(x)=\frac{1}{v_{1}} \cdot \frac{\dot{x}}{\sqrt{h_{1}^{2}+x^{2}}}-\frac{1}{v_{2}} \cdot \frac{l-x}{\sqrt{h_{2}^{2}+(l-x)^{2}}}, x \in[0, l] \\
T^{\prime \prime}(x)=\frac{1}{v_{1}} \cdot \frac{h_{1}^{2}}{\left(h_{1}^{2}+x^{2}\right)^{\frac{3}{2}}}+\frac{1}{v_{2}} \cdot \frac{h_{2}^{2}}{\left[h_{2}^{2}+(l-x)^{2}\right]^{\frac{3}{2}}}>0, x \in[0, l]
\end{gathered}
$$

$$
T^{\prime}(0)<0, T^{\prime}(l)>0,
$$

又 $T^{\prime}(x)$ 在 $[0, l]$ 上连续, 故 $T^{\prime}(x)$ 在 $(0, l)$ 内存在唯一零点 $x_{0}$, 且 $x_{0}$ 是 $T(x)$ 在 $(0, l)$ 内的唯一极小值点, 从而也是 $T(x)$ 在 $[0, l]$ 上的最小值点.

设 $x_{0}$ 满足 $T^{\prime}(x)=0$, 即

$$
\frac{x_{0}}{v_{1} \sqrt{h_{1}^{2}+x_{01}^{2}}}=\frac{l-x_{0}}{v_{2} \sqrt{h_{2}^{2}+\left(l-x_{01}\right)^{2}}} \text {. }
$$

记

$$
\frac{x_{0}}{\sqrt{h_{1}^{2}+x_{0}^{2}}}=\sin \theta_{1}, \frac{l-x_{11}}{\sqrt{h_{2}^{2}+\left(l-x_{11}\right)^{2}}}=\sin \theta_{2},
$$

就得到

$$
\frac{\sin \theta_{1}}{v_{1}}=\frac{\sin \theta_{2}}{v_{2}} .
$$

这就是说, 当 $P$ 点满足以上条件时, $A P B$ 就是光线的传播路径. 上式就是光学 中著名的折射定律,其中 $\theta_{1}, \theta_{2}$ 分别是光线的人射角和折射角 (见图 3-18).

还要指出,实际问题中，往往根据问题的性质就可以断定可导函数 $f(x)$ 确 有最大值或最小值，而且一定在定义区间内部取得. 这时如果 $f(x)$ 在定义区间 内部只有一个驻点 $x_{01}$, 那么不必讨论 $f\left(x_{0}\right)$ 是不是极值, 就可以断定 $f\left(x_{0}\right)$ 是 最大值或最小值.

例 6 把一根直径为 $d$ 的圆木锯成截面为矩形的梁 (图3-19). 问矩形截面的高 $h$ 和宽 $b$ 应如何选择才能使梁 的抗弯截面模量最大?

解 由力学分析知道: 矩形梁的抗弯截面模量为

$$
W=\frac{1}{6} b h^{2} \text {. }
$$

由图 3-19看出, $b$ 与 $h$ 有下面的关系:

$$
h^{2}=d^{2}-b^{2} \text {, }
$$

国 $3-19$

因而

$$
W=\frac{1}{6}\left(d^{2} b-b^{3}\right) \text {. }
$$

这样, $W$ 就与 $b$ 存在函数关系, $b$ 的变化范围是 $(0, d)$. 现在, 问题化为 $: b$ 等于 多少时目标函数 $W=W(b)$ 取最大值? 为此, 求 $W$ 对 $b$ 的导数：

$$
W^{\prime}=\frac{1}{6}\left(d^{2}-3 b^{2}\right) \text {. }
$$

令 $W^{\prime}=0$, 解得

$$
b=\sqrt{\frac{1}{3}} d
$$

由于梁的最大抗弯截面模量一定存在, 而且在 $(0, d)$ 内部取得; 现在, $W^{\prime}=0$ 在 $(0, d)$ 内只有一个根 $b=\sqrt{\frac{1}{3}} d$, 所以, 当 $b=\sqrt{\frac{1}{3}} d$ 时, $W$ 的值最大. 这时,

$$
h^{2}=d^{2}-b^{2}=d^{2}-\frac{1}{3} d^{2}=\frac{2}{3} d^{2} \text {, }
$$

即

$$
\begin{gathered}
h=\sqrt{\frac{2}{3}} d . \\
d: h: b=\sqrt{3}: \sqrt{2}: 1 .
\end{gathered}
$$

例 7 假设某工厂生产某产品 $x$ 千件的成本是 $c(x)=x^{3}-6 x^{2}+15 x$, 售 出该产品 $x$ 千件的收入是 $r(x)=9 x$. 问是否存在一个能取得最大利润的生产 水平? 如果存在的话, 找出这个生产水平.

解 由题意知, 售出 $x$ 千件产品的利润是

$$
p(x)=r(x)-c(x) .
$$

如果 $p(x)$ 取得最大值, 那么它一定在使得 $p^{\prime}(x)=0$ 的生产水平处获得. 因此, 令

即

$$
p^{\prime}(x)=r^{\prime}(x)-c^{\prime}(x)=0,
$$

得

$$
r^{\prime}(x)=c^{\prime}(x) \text {. }
$$
$$

x^{2}-4 x+2=0 \text {. }

$$

解得 $x=\frac{4 \pm \sqrt{8}}{2}=2 \pm \sqrt{2}, x_{1}=2-\sqrt{2} \approx 0.586, x_{2}=2+\sqrt{2} \approx 3.414$.

又 $p^{\prime \prime}(x)=-6 x+12, p^{\prime \prime}\left(x_{1}\right)>0, p^{\prime \prime}\left(x_{2}\right)<0$.

故在 $x_{2}=3.414$ 处达到最大利润, 而在 $x_{1}=0.586$ 处发生局部最大亏损.

在经济学中, 称 $c^{\prime}(x)$ 为边际成本, $r^{\prime}(x)$ 为边 际收入, $p^{\prime}(x)$ 为边际利润. 上述结果表明: 在给出最 大利润的生产水平上, $r^{\prime}(x)=c^{\prime}(x)$, 即边际收入等 于边际成本.上面的结果也可以从图 3-20 的成本。 曲线和收入曲线中看出.

## 习 题 3-5

1. 求下列函数的极值:
(1) $y=2 x^{3}-6 x^{2}-18 x+7$;
(2) $y=x-\ln (1+x)$;
(3) $y=-x^{4}+2 x^{2}$;
(4) $y=x+\sqrt{1-x}$;
(5) $y=\frac{1+3 x}{\sqrt{4+5 x^{2}}}$;
(6) $y=\frac{3 x^{2}+4 x+4}{x^{2}+x+1}$
(7) $y=\mathrm{e}^{x} \cos x$;
(8) $y=x^{\frac{1}{x}}$;
(9) $y=3-2(x+1)^{\frac{1}{3}}$;
(10) $y=x+\tan x$.
2. 试证明: 如果函数 $y=a x^{3}+b x^{2}+c x+d$ 满足条件 $b^{2}-3 a c<0$, 那么这函数没有 极值.
3. 试问 $a$ 为何值时, 函数 $f(x)=a \sin x+\frac{1}{3} \sin 3 x$ 在 $x=\frac{\pi}{3}$ 处取得极值? 它是极大值 还是极小值? 并求此极值.
4. 求下列函数的熶大值、最小值;
(1) $y=2 x^{3}-3 x^{2},-1 \leqslant x \leqslant 4$;
(2) $y=x^{4}-8 x^{2}+2,-1 \leqslant x \leqslant 3$;
(3) $y=x+\sqrt{1-x},-5 \leqslant x \leqslant 1$.
5. 问函数 $y=2 x^{3}-6 x^{2}-18 x-7(1 \leqslant x \leqslant 4)$ 在何处取得最大值? 并求出它的最大值.
6. 问函数 $y=x^{2}-\frac{54}{x}(x<0)$ 在何处取得报小值?
7. 问函数 $y=\frac{x}{x^{2}+1}(x \geqslant 0)$ 在何处取得报大值?
8. 某车间靠墙壁姴盖一间长方形小屋, 现有存砖只的砌 $20 \mathrm{~m}$ 长的墙壁. 问应围成怎样 的长方形才能使这间小屋的面积樶大?
9. 要造一圆柱形油罐, 体积为 $V$, 问底半径 $r$ 和高h各等于多少时, 才能使表面积樶小? 这时底直径与高的比是多少?
10. 某地区防空洞的截面拟建成矩形加半圆 (图 3-21). 截面的面积为 $5 \mathrm{~m}^{2}$. 问底宽 $x$ 为多少时才能使截面的周长报小, 从而使建造时所用的材料最省?

国 3-21

11. 设有质量为 $5 \mathrm{~kg}$ 的物体, 粗于水平面上, 受力 $F$ 的作用而开始移动 (图 3-22). 设磨擦系数 $\mu=0.25$, 问力 $F$ 与水平线的交角 $a$ 为多少时,才可使力 $F$ 的大小为佷小.
12. 有一杜枺, 支点在它的一端. 在距支点 $0.1 \mathrm{~m}$ 处挂一质甪为 $49 \mathrm{~kg}$ 的物体. 加力于杜 杆的另一端使杜杆保持水平 (图 3-23). 如果杜杆的线密度为 $5 \mathrm{~kg} / \mathrm{m}$, 求樶省力的杆长?
13. 从一块半径为 $R$ 的圆铁片上挖去一个阘形做成一个霃斗(图 3-24). 问留下的阙形 的中心角 $\varphi$ 取多大时, 做成的洞斗的容积最大?

14. 某吊车的车身高为 $1.5 \mathrm{~m}$, 吊傽长 $15 \mathrm{~m}$. 现在要把一个 $6 \mathrm{~m}$ 宽、2 $\mathrm{m}$ 高的屋架, 水平地 吊到 $6 \mathrm{~m}$ 高的柱子上去(图 3-25), 问能否吊得上去?

国 3-25

15. 一房地产公司有 50 套公寓要出租. 当月租金定为 1000 元时,公寓会全部租出去. 当 月租金每增加 50 元时,就会多一套公寓租不出去,而租出去的公离每月需花费 100 元的维修 费。试问房租定为多少可获得最大收入?
16. 已知制作一个背包的成本为 40 元. 如果每一个背包的售出价为 $x$ 元.售出的背包数 由

$$
n=\frac{a}{x-40}+b(80-x)
$$

给出,其中 $a, b$ 为正常数. 问什么样的售出价格能带来最大利润?

## 第六节 函数图形的描绘

借助于一阶导数的符号, 可以确定函数图形在哪个区间上上升, 在哪个区间 上下降,在什么地方有极值点; 借助于二阶导数的符号, 可以确定函数图形在哪 个区间上为凹,在哪个区间上为凸，在什么地方有拐点. 知道了函数图形的升降、 凹凸以及极值点和拐点后, 也就可以掌握函数的性态,并把函数的图形画得比较 准确. 现在, 随着现代计算机技术的发展, 借助于计算机和许多数学软件, 可以方 便地画出各种函数的图形. 但是, 如何识别机器作图中的误差, 如何掌握图形上 的关链点, 如何选择作图的范围等, 从而进行人工干预, 仍然需要我们有运用微 分学的方法描绘函数图形的基本知识.

利用导数描绘函数图形的一般步骤如下:

第一步 确定函数 $y=f(x)$ 的定义域及函数所具有的某些特性(如奇偶 性、周期性等), 并求出函数的一阶导数 $f^{\prime}(x)$ 和二阶导数 $f^{\prime \prime}(x)$;

第二步 求出一阶导数 $f^{\prime}(x)$ 和二阶导数 $f^{\prime \prime}(x)$ 在函数定义域内的全部零 点, 并求出函数 $f(x)$ 的间断点及 $f^{\prime}(x)$ 和 $f^{\prime \prime}(x)$ 不存在的点, 用这些点把函数 的定义域划分成几个部分区间;

第三步 确定在这些部分区间内 $f^{\prime}(x)$ 和 $f^{\prime \prime}(x)$ 的符号, 并由此确定函数 图形的升降和凹凸，极值点和拐点;

第四步 确定函数图形的水平、铅直渐近线以及其他变化趋势;

第五步 算出 $f^{\prime}(x)$ 和 $f^{\prime \prime}(x)$ 的零点以及不存在的点所对应的函数值, 定 出图形上相应的点; 为了把图形描绘得准确些, 有时还需要补充一些点; 然后结 合第三、四步中得到的结果, 联结这些点西出函数 $y=f(x)$ 的图形.

例 1 画出函数 $y=x^{3}-x^{2}-x+1$ 的图形.

解 (1) 所给函数 $y=f(x)$ 的定义域为 $(-\infty,+\infty)$, 而

$$
\begin{aligned}
& f^{\prime}(x)=3 x^{2}-2 x-1=(3 x+1)(x-1), \\
& f^{\prime \prime}(x)=6 x-2=2(3 x-1) .
\end{aligned}
$$

(2) $f^{\prime}(x)$ 的零点为 $x=-\frac{1}{3}$ 和 $1 ; f^{\prime \prime}(x)$ 的零点为 $x=\frac{1}{3}$. 将点 $x=-\frac{1}{3}$, $\frac{1}{3}, 1$ 由小到大排列，依次把定义域 $(-\infty,+\infty)$ 划分成下列四个部分区间:

$$
\left(-\infty,-\frac{1}{3}\right],\left[-\frac{1}{3}, \frac{1}{3}\right],\left[\frac{1}{3}, 1\right],[1,+\infty) .
$$

(3) 在 $\left(-\infty,-\frac{1}{3}\right)$ 内, $f^{\prime}(x)>0, f^{\prime \prime}(x)<0$, 所以在 $\left(-\infty,-\frac{1}{3}\right]$ 上的曲线 弧上升而且是凸的.

在 $\left(-\frac{1}{3}, \frac{1}{3}\right)$ 内, $f^{\prime}(x)<0, f^{\prime \prime}(x)<0$, 所以在 $\left[-\frac{1}{3}, \frac{1}{3}\right]$ 上的曲线弧下 降而且是凸的.

同样, 可以讨论在区间 $\left[\frac{1}{3}, 1\right]$ 上及在区间 $[1,+\infty)$ 上相应的曲线弧的升降 和凹凸. 为了明确起见, 我们把所得的结论列成下表:

| $x$ | $\left(-\infty,-\frac{1}{3}\right)$ | $-\frac{1}{3}$ | $\left(-\frac{1}{3}, \frac{1}{3}\right)$ | $\frac{1}{3}$ | $\left(\frac{1}{3}, 1\right)$ | 1 | $(1,+\infty)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $f(x)$ | + | 0 | - | - | - | 0 | + |
| $f^{\prime \prime}(x)$ | - | - | - | 0 | + | + | + |
| $\begin{array}{c}y=f(x) \\ \text { 的图形 }\end{array}$ | $r$ | 极大 | 7 | 拐点 | $\checkmark$ | 极小 | , |

这里记号$\nearrow$表示曲线弧上升而且是凸的，$\searrow$表示曲线弧下降而且是凸的，〈表示曲线弧下降而且是凹的, $’$ 表示曲线弧上升而且是凹的.

(4) 当 $x \rightarrow+\infty$ 时, $y \rightarrow+\infty$; 当 $x \rightarrow-\infty$ 时, $y \rightarrow-\infty$;

(5) 算出 $x=-\frac{1}{3}, \frac{1}{3}, 1$ 处的函数值:

$$
f\left(-\frac{1}{3}\right)=\frac{32}{27}, f\left(\frac{1}{3}\right)=\frac{16}{27}, f(1)=0 .
$$

从而得到函数 $y=x^{3}-x^{2}-x+1$ 图形上的三个点:

$$
\left(-\frac{1}{3}, \frac{32}{27}\right),\left(\frac{1}{3}, \frac{16}{27}\right),(1,0) \text {. }
$$

适当补充一些点. 例如, 计算出

$$
f(-1)=0, f(0)=1, f\left(\frac{3}{2}\right)=\frac{5}{8},
$$

就可补充描出点 $(-1,0)$, 点 $(0,1)$ 和点 $\left(\frac{3}{2}, \frac{5}{8}\right)$. 结合(3)、(4)中得到的结果, 就可以画 出

的图形(图 3-26).

$$
y=x^{3}-x^{2}-x+1
$$

例 2 描绘函数 $y=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}$ 的图形.

解 (1) 所给函数 $f(x)=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}$ 的定

义域为 $(-\infty,+\infty)$.

由于

$$
f(-x)=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{(-x)^{2}}{2}}=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}=f(x),
$$

所以 $f(x)$ 是偶函数, 它的图形关于 $y$ 轴对称. 因此可以只讨论 $[0,+\infty)$ 上该函 数的图形. 求出

$$
\begin{aligned}
f^{\prime}(x) & =\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}} \cdot(-x)=-\frac{1}{\sqrt{2 \pi}} x \mathrm{e}^{-\frac{x^{2}}{2}}, \\
f^{\prime \prime}(x) & =-\frac{1}{\sqrt{2 \pi}}\left[\mathrm{e}^{-\frac{x^{2}}{2}}+x \mathrm{e}^{-\frac{x^{2}}{2}} \cdot(-x)\right] \\
& =\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}\left(x^{2}-1\right) .
\end{aligned}
$$

(2) 在 $[0,+\infty)$ 上, $f^{\prime}(x)$ 的零点为 $x=0 ; f^{\prime \prime}(x)$ 的零点为 $x=1$. 用点 $x=1$ 把 $[0,+\infty)$ 划分成两个区间 $[0,1]$ 和 $[1,+\infty)$.

(3) 在 $(0,1)$ 内, $f^{\prime}(x)<0, f^{\prime \prime}(x)<0$, 所以在 $[0,1]$ 上的曲线弧下降而且是 凸的. 结合 $f^{\prime}(0)=0$ 以及图形关于 $y$ 轴对称可知, $x=0$ 处函数 $f(x)$ 有极大值.

在 $(1,+\infty)$ 内, $f^{\prime}(x)<0, f^{\prime \prime}(x)>0$, 所以在 $[1,+\infty)$ 上的曲线弧下降而且 是凹的.

上述的这些结果, 可以列成下表：

| $x$ | 0 | $(0,1)$ | 1 | $(1,+\infty)$ |
| :---: | :---: | :---: | :---: | :---: |
| $f^{\prime}(x)$ | 0 | - | - | - |
| $f^{\prime \prime}(x)$ | - | - | 0 | + |
| $y=f(x)$ 的图形 | 极大 | - | 拐点 | ( |

(4) 由于 $\lim _{x \rightarrow+\infty} f(x)=0$, 所以图形有一条水平渐近线 $y=0$.

(5) 算出 $f(0)=\frac{1}{\sqrt{2 \pi}}, f(1)=\frac{1}{\sqrt{2 \pi \mathrm{e}}}$. 从而得到函数

$$
y=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}
$$

图形上的两点 $M_{1}\left(0, \frac{1}{\sqrt{2 \pi}}\right)$ 和 $M_{2}\left(1, \frac{1}{\sqrt{2 \pi \mathrm{e}}}\right)$. 又由 $f(2)=\frac{1}{\sqrt{2 \pi \mathrm{e}^{2}}}$

得 $M_{3}\left(2, \frac{1}{\sqrt{2 \pi} \mathrm{e}^{2}}\right)$. 结合 (3)、(4) 的讨论, 画出函数 $y=\frac{1}{\sqrt{2 \pi}} \mathrm{e}^{-\frac{x^{2}}{2}}$ 在 $[0,+\infty)$ 上的 图形.最后,利用图形的对称性,便可得到函数在 $(-\infty, 0]$ 上的图形(图 3-27).

解 (1) 所给函数 $y=f(x)$ 的定义域为 $(-\infty,-3),(-3,+\infty)$.

$$
f^{\prime}(x)=\frac{36(3-x)}{(x+3)^{3}}, f^{\prime \prime}(x)=\frac{72(x-6)}{(x+3)^{4}} \text {. }
$$

(2) $f^{\prime}(x)$ 的零点为 $x=3 ; f^{\prime \prime}(x)$ 的零点为 $x=6 ; x=-3$ 是函数的间断点. 点 $x=-3 、 x=3$ 和 $x=6$ 把定义域划分成四个部分区间:

$$
(-\infty,-3),(-3,3],[3,6],[6,+\infty) \text {. }
$$

(3) 在各部分区间内 $f^{\prime \prime}(x)$ 及 $f^{\prime \prime}(x)$ 的符号、相应曲线弧的升降及凹凸、极 值点和拐点等如下表:

| $x$ | $(-\infty,-3)$ | $(-3,3)$ | 3 | $(3,6)$ | 6 | $(6,+\infty)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $f^{\prime}(x)$ | - | + | 0 | - | - | - |
| $f^{\prime \prime}(x)$ | - | - | - | - | 0 | + |
| $\begin{array}{c}y=f(x) \\ \text { 的图形 }\end{array}$ | $\nearrow$ | $r$ | 极大 | 7 | 拐点 | ( |

(4) 由于 $\lim _{x \rightarrow \infty} f(x)=1, \lim _{x \rightarrow-3} f(x)=-\infty$, 所以图形有一条水平渐近线 $y=1$ 和一条铅直渐近线 $x=-3$.

(5) 算出 $x=3,6$ 处的函数值:

$$
f(3)=4, f^{\prime}(6)=\frac{11}{3},
$$

从而得到图形上的两个点:

$$
M_{1}(3,4), M_{2}\left(6, \frac{11}{3}\right)
$$

又由于

$$
f(0)=1, f(-1)=-8, f(-9)=-8, f(-15)=-\frac{11}{4},
$$

得图形上的四个点:

$$
\begin{array}{ll}
M_{3}(0,1), & M_{4}(-1,-8), \\
M_{5}(-9,-8), & M_{6}\left(-15,-\frac{11}{4}\right) .
\end{array}
$$

结合 (3)、(4) 中得到的结果, 画出函数 $y=1+\frac{36 x}{(x+3)^{2}}$ 的图形如图 3-28 所示.

## 习 题 3-6

描绘下列函数的图形:

1. $y=\frac{1}{5}\left(x^{4}-6 x^{2}+8 x+7\right)$;
2. $y=\frac{x}{1+x^{2}}$;
3. $y=\mathrm{e}^{-(x-1)^{2}}$;
4. $y=x^{2}+\frac{1}{x}$;
5. $y=\frac{\cos x}{\cos 2 x}$.

## 第七节 曲 率

## 一、弧微分

作为曲率的预备知识, 先介绍弧微分的概念.

设函数 $f(x)$ 在区间 $(a, b)$ 内具有连续导数. 在曲线 $y=f(x)$ 上取固定点 $M_{0}\left(x_{0}, y_{0}\right)$ 作为度量弧长的基点 (图 3-29), 并规定依 $x$ 增大的方向作为曲线 的正向. 对曲线上任一点 $M(x, y)$, 规定有向弧段 $\overparen{M_{0} M}$ 的值 $s$ (简称为弧 $s$ ) 如 下: $s$ 的绝对值等于这弧段的长度, 当有向弧段 $\overparen{M_{0} M}$ 的方向与曲线的正向一致

(1) 有向弧段 $\overparen{M_{11} M}$ 的值也常记作 $\overparen{M_{0} M}$, 即记费 $\overparen{M_{11} M}$ 既表示有向弧段，又表示有向硍段的值. 时 $s>0$, 相反时 $s<0$. 显然, 弧 $s$ 与 $x$ 存在函数关系: $s=s(x)$, 而且 $s(x)$ 是 $x$ 的单调增加函数.下面来求 $s(x)$ 的导数及微分.

设 $x, x+\Delta x$ 为 $(a, b)$ 内两个邻近的点,它们在 曲线 $y=f(x)$ 上的对应点为 $M, M^{\prime}$ (图 3-29), 并 设对应于 $x$ 的增量 $\Delta x$, 弧 $s$ 的增旺为 $\Delta s$, 那么

$$
\Delta s=\widehat{M_{0} M^{\prime}}-\widehat{M_{0} M}=\widehat{M M}^{\prime} \text {. }
$$

于是 $\left(\frac{\Delta s}{\Delta x}\right)^{2}=\left(\frac{\widehat{M M^{\prime}}}{\Delta x}\right)^{2}=\left(\frac{\widehat{M M^{\prime}}}{\left|M M^{\prime}\right|}\right)^{2} \cdot \frac{\left|M M^{\prime}\right|^{2}}{(\Delta x)^{2}}$.

$$
\begin{aligned}
& =\left(\frac{\widehat{M M^{\prime}}}{\left|M M^{\prime}\right|}\right)^{2} \cdot \frac{(\Delta x)^{2}+(\Delta y)^{2}}{(\Delta x)^{2}} \\
& =\left(\frac{\widehat{M M^{\prime}}}{\left|M M^{\prime}\right|}\right)^{2}\left[1+\left(\frac{\Delta y}{\Delta x}\right)^{2}\right]
\end{aligned}
$$

$$
\frac{\Delta s}{\Delta x}= \pm \sqrt{\left(\frac{\widehat{M M^{\prime}}}{\left|M M^{\prime}\right|}\right)^{2} \cdot\left[1+\left(\frac{\Delta y}{\Delta x}\right)^{2}\right]} .
$$

令 $\Delta x \rightarrow 0$ 取极限, 由于 $\Delta x \rightarrow 0$ 时, $M^{\prime} \rightarrow M$, 这时弧的长度与弦的长度之比的极 限等于 1 , 即

$$
\lim _{M^{\prime} \rightarrow M} \frac{\left|\widehat{M M^{\prime}}\right|}{\left|M M^{\prime}\right|}=1
$$

又

$$
\lim _{\Delta x \rightarrow 0} \frac{\Delta y}{\Delta x}=y^{\prime}
$$

因此得

$$
\frac{\mathrm{d} s}{\mathrm{~d} x}= \pm \sqrt{1+y^{\prime 2}} \text {. }
$$

由于 $s=s(x)$ 是单调增加函数, 从而根号前应取正号, 于是有

$$
\mathrm{d} s=\sqrt{1+y^{\prime 2}} \mathrm{~d} x \text {. }
$$

这就是弧微分公式.

## 二、曲率及其计算公式

我们直觉地认识到:直线不弯曲, 半径较小的圆弯曲得比半径较大的圆厉害 些, 而其他曲线的不同部分有不同的弯曲程度,例如抛物线 $y=x^{2}$ 在顶点附近 弯曲得比远离顶点的部分厉害些.

在工程技术中,有时需要研究曲线的弯曲程度.例如,船体结构中的钢梁, 机 床的转轴等, 它们在荷载作用下要产生弯曲变形, 在设计时对它们的弯曲必须有 一定的限制, 这就要定量地研究它们的弯曲程度. 为此首先要讨论如何用数量来 描述曲线的弯曲程度.

在图 3-30 中可以看出, 弧段 $\widehat{M}_{1} M_{2}$ 比较平直,当动点沿这段弧从 $M_{1}$ 移动 到 $M_{2}$ 时, 切线转过的角度 $\varphi_{1}$ 不大, 而弧段 $\widehat{M}_{2} M_{3}$ 弯曲得比较厉害, 角 $\varphi_{2}$ 就比 较大.

但是,切线转过的角度的大小还不能完全反映曲线弯曲的程度. 例如, 从图 3-31 中可以看出，两段曲线弧 $\widehat{M}_{1} M_{2}$ 及 $\widehat{N_{1} N_{2}}$ 尽管切线转过的角度都是 $\varphi$, 然 而弯曲程度并不相同,短弧段比长弧段弯曲得厉害些. 由此可见, 曲线弧的弯曲 程度还与弧段的长度有关.

按上面的分析, 我们引人描述曲线弯曲程度的曲率概念如下.

设曲线 $C$ 是光滑的 $\left(\mathbb{D}\right.$, 在曲线 $C$ 上选定一点 $M_{0}$ 作为度量弧 $s$ 的基点. 设曲 线上点 $M$ 对应于弧 $s$, 在点 $M$ 处切线的倾角为 $\alpha$ (这里假定曲线 $C$ 所在的平面 上已设立了 $x O y$ 坐标系)，曲线上另外一点 $M^{\prime}$ 对应 于弧 $s+\Delta s$, 在点 $M^{\prime}$ 处切线的倾角为 $\alpha+\Delta \alpha$ (图 332 ), 那么, 弧段 $\widehat{M M}^{\prime}$ 的长度为 $|\Delta s|$, 当动点从 $M$ 移 动到 $M^{\prime}$ 时切线转过的角度为 $|\Delta \alpha|$.

我们用比值 $\frac{|\Delta a|}{|\Delta s|}$, 即单位弧段上切线转过的角 度的大小来表达弧段 $\widehat{M M}^{\prime}$ 的平均弯曲程度, 把这比 值叫做弧段 $\widehat{M M^{\prime}}$ 的平均曲率, 并记作 $\bar{K}$, 即

$$
\bar{K}=\left|\frac{\Delta \alpha}{\Delta s}\right| \text {. }
$$

类似于从平均速度引进瞬时速度的方法, 当 $\Delta s \rightarrow 0$ 时 (即 $M^{\prime} \rightarrow M$ 时), 上 述平均曲率的极限叫做曲线 $C$ 在点 $M$ 处的曲率, 记作 $K$, 即

$$
K=\lim _{\Delta x \rightarrow 0}\left|\frac{\Delta \alpha}{\Delta s}\right| .
$$

在 $\lim _{\Delta x \rightarrow 0} \frac{\Delta \alpha}{\Delta s}=\frac{\mathrm{d} \alpha}{\mathrm{d} s}$ 存在的条件下, $K$ 也可以表示为

$$
K=\left|\frac{\mathrm{d} \alpha}{\mathrm{d} s}\right| \text {. }
$$

对于直线来说, 切线与直线本身重合, 当点沿直线移动时, 切线的倾角 $\alpha$ 不 变(图 3-33), $\Delta \alpha=0, \frac{\Delta \alpha}{\Delta s}=0$, 从而 $K=\left|\frac{\mathrm{d} \alpha}{\mathrm{d} s}\right|=0$. 这就是说, 直线上任意点 $M$ 处的曲率都等于零, 这与我们直觉认识到的“直线不弯曲”一致.

设圆的半径为 $a$, 由图 3-34 可见圆在点 $M 、 M^{\prime}$ 处的切线所夹的角 $\Delta a$ 等 于中心角 $M D M^{\prime}$. 但 $\angle M D M^{\prime}=\frac{\Delta s}{a}$, 于是

从而

$$
\begin{gathered}
\frac{\Delta \alpha}{\Delta s}=\frac{\frac{\Delta s}{a}}{\Delta s}=\frac{1}{a}, \\
K=\left|\frac{\mathrm{d} \alpha}{\mathrm{d} s}\right|=\frac{1}{a} .
\end{gathered}
$$

图 $3-\mathbf{3 3}$

因为点 $M$ 是圆上任意取定的一点, 上述结论表示圆上各点处的曲率都等于半 径 $a$ 的倒数 $\frac{1}{a}$, 这就是说, 圆的弯曲程度到处一样, 且半径越小曲率越大, 即圆 弯曲得越厉害.

在一般情况下,我们根据 (2)式来导出便于实际计算曲率的公式.

设曲线的直角坐标方程是 $y=f(x)$, 且 $f(x)$ 具有二阶导数 (这时 $f^{\prime}(x)$ 连 续, 从而曲线是光滑的). 因为 $\tan \alpha=y^{\prime}$, 所以

$$
\begin{gathered}
\sec ^{2} \alpha \frac{\mathrm{d} \alpha}{\mathrm{d} x}=y^{\prime \prime}, \\
\frac{\mathrm{d} \alpha}{\mathrm{d} x}=\frac{y^{\prime \prime}}{1+\tan ^{2} \alpha}=\frac{y^{\prime \prime}}{1+y^{\prime 2}},
\end{gathered}
$$

于是

$$
\mathrm{d} \alpha=\frac{y^{\prime \prime}}{1+y^{\prime 2}} \mathrm{~d} x \text {. }
$$

又由 (1)知道

$$
\mathrm{d} s=\sqrt{1+y^{\prime 2}} \mathrm{~d} x .
$$

从而, 根据曲率 $K$ 的表达式(2),有

$$
K=\frac{\left|y^{\prime \prime}\right|}{\left(1+y^{\prime 2}\right)^{3 / 2}}
$$

设曲线由参数方程

$$
\mid \begin{aligned}
& x=\varphi(t), \\
& y=\phi(t)
\end{aligned}
$$

给出, 则可利用由参数方程所确定的函数的求导法, 求出 $y^{\prime}{ }_{.}$及 $y^{\prime \prime}{ }_{x}$, 代入 (3) 便 得

$$
K=\frac{\left|\varphi^{\prime}(t) \psi^{\prime \prime}(t)-\varphi^{\prime \prime}(t) \psi^{\prime}(t)\right|}{\left[\varphi^{\prime 2}(t)+\psi^{\prime 2}(t)\right]^{3 / 2}} .
$$

例 1 计算等边双曲线 $x y=1$ 在点 $(1,1)$ 处的曲率.

解 由 $y=\frac{1}{x}$, 得

$$
y^{\prime}=-\frac{1}{x^{2}}, y^{\prime \prime}=\frac{2}{x^{3}} .
$$

因此,

$$
\left.y^{\prime}\right|_{x=1}=-1,\left.y^{\prime \prime}\right|_{x=1}=2 \text {. }
$$

把它们代入公式 (3), 便得曲线 $x y=1$ 在点 $(1,1)$ 处的曲率为

$$
K=\frac{2}{\left[1+(-1)^{2}\right]^{3 / 2}}=\frac{\sqrt{2}}{2} .
$$

例 2 抛物线 $y=c x^{2}+b x+c$ 上哪一点处的曲率最大?

解 由 $y=a x^{2}+b x+c$, 得

$$
y^{\prime}=2 a x+b, y^{\prime \prime}=2 a,
$$

代入公式(3),得

$$
K=\frac{|2 a|}{\left[1+(2 a x+b)^{2}\right]^{3 / 2}} .
$$

因为 $K$ 的分子是常数 $|2 a|$, 所以只要分母最小, $K$ 就最大. 容易看出, 当 $2 a x+b=0$, 即 $x=-\frac{b}{2 a}$ 时, $K$ 的分母最小, 因而 $K$ 有最大值 $|2 a|$. 而 $x=-\frac{b}{2 a}$ 所对应的点为抛物线的顶点. 因此,抛物线在顶点处的曲率最大.

在有些实际问题中, $\left|y^{\prime}\right|$ 同 1 比较起来是很小的(有的工程技术书上把这种 关系记成 $\left.\left|y^{\prime}\right| \ll 1\right)$, 可以忽略不计. 这时, 由

$$
1+y^{\prime 2} \approx 1 \text {, }
$$

而有曲率的近似计算公式

$$
\text { - } K=\frac{\left|y^{\prime \prime}\right|}{\left(1+y^{\prime 2}\right)^{3 / 2}} \approx\left|y^{\prime \prime}\right| \text {. }
$$

这就是说, 当 $\left|y^{\prime}\right| \ll 1$ 时, 曲率 $K$ 近似于 $\left|y^{\prime \prime}\right|$. 经过这样简化之后, 对一些复杂 问题的计算和讨论就方便多了.

## 三、曲率圆与曲率半径

设曲线 $y=f(x)$ 在点 $M(x, y)$ 处的曲率为 $K(K \neq 0)$. 在点 $M$ 处的曲线 的法线上, 在凹的一侧取一点 $D$, 使 $|D M|=\frac{1}{K}=\rho$. 以 $D$ 为圆心, $\rho$ 为半径作圆 (图 3-35), 这个圆叫做曲 线在点 $M$ 处的曲率圆, 曲率圆的圆心 $D$ 叫做曲线在 点 $M$ 处的曲率中心, 曲率圆的半径 $\rho$ 叫做曲线在点 $M$ 处的曲率半径.

按上述规定可知, 曲率圆与曲线在点 $M$ 有相同 的切线和曲率, 且在点 $M$ 邻近有相同的凹向. 因此, 在实际问题中, 常常用曲率圆在点 $M$ 邻近的一段圆

按上述规定. 曲线在点 $M$ 处的曲率 $K(K \neq 0)$ 与曲线在点 $M$ 处的曲率半 径 $\rho$ 有如下关系:

$$
\rho=\frac{1}{K}, K=\frac{1}{\rho} .
$$

这就是说: 曲线上一点处的曲率半径与曲线在该点 处的曲率互为倒数.

例 3 设工件内表面的截线为抛物线 $y=$ $0.4 x^{2}$ (图 3-36). 现在要用砂轮磨削其内表面. 问用直径多大的砂轮才比较合适?

解 为了在磨削时不使砂轮与工件接触处附 近的那部分工件磨去太多，砂轮的半径应不大于扰 物线上各点处曲率半径中的最小值. 由本节例 2 知

而有

$$
y^{\prime}=0.8 x, y^{\prime \prime}=0.8 \text {, }
$$

把它们代入公式(3), 得

$$
\left.y^{\prime}\right|_{x=0}=0,\left.y^{\prime \prime}\right|_{x=0}=0.8 \text {. }
$$

$$
K=0.8 .
$$

因而求得抛物线顶点处的曲率半径

$$
\rho=\frac{1}{K}=1.25 \text {. }
$$

所以选用砂轮的半径不得超过 1.25 单位长, 即直径不得超过 2.50 单位长.

对于用砂轮磨削一般工件的内表面时, 也有类似的结论, 即选用的砂轮的半 径不应超过这工件内表面的截线上各点处曲率半径中的最小值.

## *四、曲率中心的计算公式 渐㞑线与渐伸线

设已知曲线的方程是 $y=f(x)$, 且其二阶导数 $y^{\prime \prime}$ 在点 $x$ 不为零, 则曲线在 对应点 $M(x, y)$ 的曲率中心 $D(\alpha, \beta)$ 的坐标为

$$
\left\{\begin{array}{l}
\alpha=x-\frac{y^{\prime}\left(1+y^{\prime 2}\right)}{y^{\prime \prime}}, \\
\beta=y+\frac{1+y^{\prime 2}}{y^{\prime \prime}} .
\end{array}\right.
$$

这是因为, 曲线 $y=f(x)$ 在点 $M(x, y)$ 的曲率圆的方程为

$$
(\xi-\alpha)^{2}+(\eta-\beta)^{2}=\rho^{2},
$$

其中 $\xi, \eta$ 是曲率圆上的动点坐标, 且

$$
\rho^{2}=\frac{1}{K^{2}}=\frac{\left(1+y^{\prime 2}\right)^{3}}{y^{\prime \prime 2}} .
$$

因为点 $M$ 在曲率圆上,所以

$$
(x-\alpha)^{2}+(y-\beta)^{2}=\rho^{2} ;
$$

又因为曲线在点 $M$ 的切线与曲率圆的半径 $D M$ 相垂直 (图 3-35), 所以

$$
y^{\prime}=-\frac{x-\alpha}{y-\beta} \text {. }
$$

由 (6)和(7)消去 $x-\alpha$,解出

$$
(y-\beta)^{2}=\frac{\rho^{2}}{1+y^{\prime 2}}=\frac{\left(1+y^{\prime 2}\right)^{2}}{y^{\prime 2}} .
$$

由于当 $y^{\prime \prime}>0$ 时曲线为凹弧, $y-\beta<0$; 当 $y^{\prime \prime}<0$ 时曲线为凸弧, $y-\beta>0$. 总之, $y^{\prime \prime}$ 与 $y-\beta$ 异号. 因此取上式两边的平方根,得

$$
y-\beta=-\frac{1+y^{\prime 2}}{y^{\prime \prime}},
$$

又

$$
x-\alpha=-y^{\prime}(y-\beta)=\frac{y^{\prime}\left(1+y^{\prime 2}\right)}{y^{\prime \prime}} .
$$

从而有公式(5). 当点 $(x, f(x))$ 沿曲线 $C$ 移动时, 相应的曲率中心 $D$ 的轨迹曲线 $G$ 称为曲 线 $C$ 的渐屈线, 而曲线 $C$ 称为曲线 $G$ 的渐伸线 (图 $3-37)$. 所以曲线 $y=f(x)$ 的渐屈线的参数方程为

$$
\left\{\begin{array}{l}
\alpha=x-\frac{y^{\prime}\left(1+y^{\prime 2}\right)}{y^{\prime \prime}}, \\
\beta=y+\frac{1+y^{\prime 2}}{y^{\prime \prime}},
\end{array}\right.
$$

其中 $y=f(x), y^{\prime}=f^{\prime}(x), y^{\prime \prime}=f^{\prime \prime}(x), x$ 为参数, 直角坐标系 $\alpha O \beta$ 与 $x O y$ 坐标系重合.

例 4 求摆线

$$
\left\{\begin{array}{l}
x=a(t-\sin t) \\
y=a(1-\cos t)
\end{array}\right.
$$

的渐屈线方程.

$$
\text { 解 } \frac{\mathrm{d} x}{\mathrm{~d} t}=a(1-\cos t), \frac{\mathrm{d} y}{\mathrm{~d} t}=a \sin t \text {, 所以 }
$$

$$
\begin{aligned}
\frac{\mathrm{d} y}{\mathrm{~d} x} & =\frac{\sin t}{1-\cos t}, \\
\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}} & =\frac{\frac{\mathrm{d}}{\mathrm{d} t}\left(\frac{\mathrm{d} y}{\mathrm{~d} x}\right)}{\frac{\mathrm{d} x}{\mathrm{~d} t}} \\
& =\frac{\frac{\cos t-1}{(1-\cos t)^{2}}}{a(1-\cos t)}=-\frac{1}{a(1-\cos t)^{2}} .
\end{aligned}
$$

将这些结果代入 (8)式并化简, 便得摆线的渐屈线的参数方程

$$
\left\{\begin{array}{l}
\alpha=a(t+\sin t), \\
\beta=a(\cos t-1),
\end{array}\right.
$$

其中 $t$ 为参数, 直角坐标系 $\alpha O \beta$ 与 $x O y$ 坐标系重合. 为了作出渐届线 (9), 令 $t=\pi+\tau$, 代入 $(9)$ 式得

$$
\left\{\begin{array}{l}
\alpha-\pi a=a(\tau-\sin \tau), \\
\beta+2 a=a(1-\cos \tau),
\end{array}\right.
$$

再令 $\alpha-\pi a=\xi, \beta+2 a=\eta$, 则得

$$
\left\{\begin{array}{l}
\xi=a(\tau-\sin \tau), \\
\eta=a(1-\cos \tau) .
\end{array}\right.
$$

在新坐标系 $\xi O_{1} \eta$ 中, 曲线 (10) 为一摆线, 其中新坐标系 $\xi O, \eta$ 由旧坐标系 $x O y$ 平移到新原点 $O_{1}(\pi a,-2 a)$ 得到. 由此可知摆线的渐屈线仍为一摆线, 如图 3-38 所示.

## 习 题 3-7

1. 求椭圆 $4 x^{2}+y^{2}=4$ 在点 $(0,2)$ 处的曲率.
2. 求曲线 $y=\ln \sec x$ 在点 $(x, y)$ 处的曲率及曲率半径.
3. 求势物线 $y=x^{2}-4 x+3$ 在其顶点处的曲率及曲率半径.
4. 求曲线 $x=a \cos ^{3} t, y=a \sin ^{3} t$ 在 $t=t_{0}$ 相应的点处的曲率.
5. 对数曲线 $y=\ln x$ 上哪一点处的曲率半径最小? 求出该点处的曲率半径.
6. 证明曲线 $y=a \operatorname{ch} \frac{x}{a}$ 在点 $(x, y)$ 处的曲率半径为 $\frac{y^{2}}{a}$.
7. 一飞机沿拖物线路径 $y=\frac{x^{2}}{10000}(y$ 轴铅直向上,单位为 $\mathrm{m}$ ) 作俯冲飞行. 在坐标原点 $O$ 处飞机的速度为 $v=200 \mathrm{~m} / \mathrm{s}$. 飞行员体重 $G=70 \mathrm{~kg}$. 求飞机倠冲至最低点即原点 $O$ 处时 坐椅对飞行员的反力.
8. 汽车连同载重共 $5 \mathrm{t}$, 在抱物线拱桥上行驶, 速度为 $21.6 \mathrm{~km} / \mathrm{h}$, 桥的跨度为 $10 \mathrm{~m}$, 拱的 矢高为 $0.25 \mathrm{~m}$ (图 3-39). 求汽车越过桥顶时对桥的压力.

9. 求曲线 $y=\ln x$ 在与 $x$ 轴交点处的曲率圆方程.

10. 求曲线 $y=\tan x$ 在点 $\left(\frac{\pi}{4}, 1\right)$ 处的曲率圆方程.

11. 求拖物线 $y^{2}=2 p x$ 的渐屈线方程.

## 第八节 方程的近似解

在科学技术问题中, 经常会遇到求解高次代数方程或其他类型的方程的问 题. 要求得这类方程的实根的精确值, 往往比较困难, 因此就需要寻求方程的近 似解.

求方程的近似解,可分两步来做.

第一步是确定根的大致范围. 具体地说, 就是确定一个区间 $[a, b]$, 使所求 的根是位于这个区间内的唯一实根。这一步工作称为根的隔离，区间 $[a, b]$ 称为 所求实根的隔离区间. 由于方程 $f(x)=0$ 的实根在几何上表示曲线 $y=f(x)$ 与 $x$ 轴交点的横坐标, 因此为了确定根的隔离㳊间, 可以先较精确地画出 $y=$ $f(x)$ 的图形, 然后从图上定出它与 $x$ 轴交点的大概位置. 由于作图和读数的误 差, 这种做法得不出根的高精确度的近似值, 但一般已可以确定出根的隔离 区间。

第二步是以根的隔离区间的端点作为根的初始近似值, 逐步改善根的近似 值的精确度,直至求得满足精确度要求的近似解. 完成这一步工作有多种方法, 这里我们介绍两种常用的方法一二分法和切线法, 按照这些方法, 编出简单的 程序, 就可以在计算机上求出方程足够精确的近似解.

## 一、二分法

设 $f(x)$ 在区间 $[a, b]$ 上连续, $f(a) \cdot f(b)<0$, 且方程 $f(x)=0$ 在 $(a ; b)$ 内 仅有一个实根 $\xi$,于是 $[a, b]$ 即是这个根的一个隔离区间.

取 $[a, b]$ 的中点 $\xi_{1}=\frac{a+b}{2}$, 计算 $f\left(\xi_{1}\right)$.

如果 $f\left(\xi_{1}\right)=0$, 那么 $\xi=\xi_{1}$ ；

如果 $f\left(\xi_{1}\right)$ 与 $f(a)$ 同号, 那么取 $a_{1}=\xi_{1}, b_{1}=b$, 由 $f\left(a_{1}\right) \cdot f\left(b_{1}\right)<0$, 即知 $a_{1}<\xi<b_{1}$, 且 $b_{1}-a_{1}=\frac{1}{2}(b-a) ;$

如果 $f\left(\xi_{1}\right)$ 与 $f(b)$ 同号, 那么取 $a_{1}=a, b_{1}=\xi_{1}$, 也有 $a_{1}<\xi<b_{1}$ 及 $b_{1}-a_{1}$ $=\frac{1}{2}(b-a)$;

总之, 当 $\xi \neq \xi_{1}$ 时, 可求得 $a_{1}<\xi<b_{1}$, 且 $b_{1}-a_{1}=\frac{1}{2}(b-a)$.

以 $\left[a_{1}, b_{1}\right]$ 作为新的隔离区间, 重复上述做法, 当 $\xi \neq \xi_{2}=\frac{1}{2}\left(a_{1}+b_{1}\right)$ 时, 可 求得 $a_{2}<\xi<b_{2}$, 且 $b_{2}-a_{2}=\frac{1}{2^{2}}(b-a)$.

如此重复 $n$ 次, 可求得 $a_{n}<\xi<b_{n}$, 且 $b_{n}-a_{n}=\frac{1}{2^{n}}(b-a)$. 由此可知, 如果 以 $a_{n}$ 或 $b_{n}$ 作为 $\xi$ 的近似值,那么其误差小于 $\frac{1}{2^{n}}(b-a)$.

例 1 用二分法求方程 $x^{3}+1.1 x^{2}+0.9 x-1.4=0$ 的实根的近似值, 使误 差不超过 $10^{-3}(\mathbb{D}$.

解 令 $f(x)=x^{3}+1.1 x^{2}+0.9 x-1.4$, 显然 $f(x)$ 在 $(-\infty,+\infty)$ 内 连续.

由 $f^{\prime}(x)=3 x^{2}+2.2 x+0.9$, 根据判别式 $B^{2}-4 A C=2.2^{2}-4 \times 3 \times 0.9=$ $-5.96<0$, 知 $f^{\prime}(x)>0$. 故 $f(x)$ 在 $(-\infty,+\infty)$ 内单调增加, $f(x)=0$ 至多有 一个实根.

由 $f(0)=-1.4<0, f(1)=1.6>0$, 知 $f(x)=0$ 在 $[0,1]$ 内有唯一的实根. 取 $a=0, b=1,[0,1]$ 即是一个隔离区间.

计算得:

$\xi_{1}=0.5, f\left(\xi_{1}\right)=-0.55<0$, 故 $a_{1}=0.5, b_{1}=1$;

$\xi_{2}=0.75, f\left(\xi_{2}\right)=0.32>0$, 故 $a_{2}=0.5, b_{2}=0.75$;

$\xi_{3}=0.625, f\left(\xi_{3}\right)=-0.16<0$, 故 $a_{3}=0.625, b_{3}=0.75$;

$\xi_{4}=0.687, f\left(\xi_{4}\right)=0.062>0$, 故 $a_{4}=0.625, b_{4}=0.687$;

$\xi_{5}=0.656, f\left(\xi_{5}\right)=-0.054<0$, 故 $a_{5}=0.656, b_{5}=0.687$;

$\xi_{6}=0.672, f\left(\xi_{6}\right)=0.005>0$, 故 $a_{6}=0.656, b_{6}=0.672$;

$\xi_{7}=0.664, f\left(\xi_{7}\right)=-0.025<0$, 故 $a_{7}=0.664, b_{7}=0.672$;

$\xi_{8}=0.668, f\left(\xi_{8}\right)=-0.010<0$, 故 $a_{8}=0.668, b_{8}=0.672$;

$\xi_{9}=0.670, f\left(\xi_{9}\right)=-0.002<0$, 故 $a_{9}=0.670, b_{9}=0.672$;

$\xi_{10}=0.671, f\left(\xi_{10}\right)=0.001>0$, 故 $a_{10}=0.670, b_{10}=0.671$.

于是

$0.670<\xi<0.671$.

即 0.670 作为根的不足近似值, 0.671 作为根的过剩近似值, 其误差都小于 $10^{-3}$.

## 二、切线法

设 $f(x)$ 在 $[a, b]$ 上具有二阶导数, $f(a) \cdot f(b)<0$ 且 $f^{\prime}(x)$ 及 $f^{\prime \prime}(x)$ 在

(1) 按本例误着不超过 $10^{-3}$ 的要求, 计算时只取 3 位小数. $[a, b]$ 上保持定号. 在上述条件下,方程 $f(x)=0$ 在 $(a, b)$ 内有唯一的实根 $\xi$, $[a, b]$ 为根的一个隔离区间. 此时, $y=f(x)$ 在 $[a, b]$ 上的图形 $\overparen{A B}$ 只有如图 3-40所示的四种不同情形.

(a) $f(a)<0, f(b)>0$ $f^{\prime}(x)>0, f^{\prime \prime}(x)>0$

(c) $f(a)<0, f(b)>0$ $f^{\prime}(x)>0, f^{\prime \prime}(x)<0$

(b) $f(a)>0, f(b)<0$ $f^{\prime}(x)<0, f^{\prime \prime}(x)>0$

(d) $f(a)>0, f(b)<0$ $f^{\prime}(x)<0, f^{\prime \prime}(x)<0$

考虑用曲线弧一端的切线来代替曲线弧, 从而求出方程实根的近似值. 这种 方法叫做切线法. 从图 3-40 中看出, 如果在纵坐标与 $f^{\prime \prime}(x)$ 同号的那个端点 (此端点记作 $\left(x_{0}, f\left(x_{0}\right)\right)$ ) 作切线, 这切线与 $x$ 轴的交点的横坐标 $x_{1}$ 就比 $x_{0}$ 更 接近方程的根 $\xi^{\Phi}$.

(1) 如图 3-41 所示, 如果把切线作在纵坐标与 $f^{\prime}(x)$ 异号的那个端点, 就不能保证切线与 $x$ 轴的交 点的模坐标 $x_{1}$ 比原来的近似值 $a$ 或 6 更接近于方程的根 $\xi$.

$$
y-f\left(x_{0}\right)=f^{\prime}\left(x_{10}\right)\left(x-x_{0}\right) .
$$

令 $y=0$, 从上式中解出 $x$, 就得到切线与 $x$ 轴交点的横坐标为

$$
x_{1}=x_{0}-\frac{f\left(x_{0}\right)}{f^{\prime}\left(x_{0}\right)} \text {, }
$$

它比 $x_{0}$ 更接近方程的根 $\xi$.

再在点 $\left(x_{1}, f\left(x_{1}\right)\right)$ 作切线, 可得根的近似值 $x_{2}$. 如此继续,一般的, 在点 $\left(x_{n-1}, f\left(x_{n-1}\right)\right)$ 作切线, 得根的近似值

$$
x_{n}=x_{n-1}-\frac{f\left(x_{n-1}\right)}{f^{\prime}\left(x_{n-1}\right)} .
$$

如果 $f(b)$ 与 $f^{\prime \prime}(x)$ 同号, 切线作在端点 $B$ (如图 3-40 情形 (a) 及 (d)), 可 记 $x_{0}=b$,仍按公式 (1)计算切线与 $x$ 轴交点的横坐标.

例 2 用切线法求方程 $x^{3}+1.1 x^{2}+0.9 x-1.4=0$ 的实根的近似值,使误 差不超过 $10^{-3}$.

解 令 $f(x)=x^{3}+1.1 x^{2}+0.9 x-1.4$. 由例 1 知 $[0,1]$ 是根的一个隔离区 间. $f(0)<0, f(1)>0$.

在 $[0,1]$ 上,

$$
\begin{aligned}
& f^{\prime}(x)=3 x^{2}+2.2 x+0.9>0, \\
& f^{\prime \prime}(x)=6 x+2.2>0,
\end{aligned}
$$

故 $f(x)$ 在 $[0,1]$ 上的图形属于图 3-40 中情形 (a). 按 $f^{\prime \prime}(x)$ 与 $f(1)$ 同号,所以 令 $x_{0}=1$.

连续应用公式(1)，得

$$
\begin{aligned}
& x_{1}=1-\frac{f(1)}{f^{\prime}(1)} \approx 0.738 ; \\
& x_{2}=0.738-\frac{f(0.738)}{f^{\prime}(0.738)} \approx 0.674 ; \\
& x_{3}=0.674-\frac{f(0.674)}{f^{\prime}(0.674)} \approx 0.671 ; \\
& x_{4}=0.671-\frac{f(0.671)}{f^{\prime}(0.671)} \approx 0.671 .
\end{aligned}
$$

至此, 计算不能再继续. 注意到 $f\left(x_{i}\right)(i=0,1, \cdots)$ 与 $f^{\prime \prime}(x)$ 同号, 知 $f(0.671)>0$, 经计算可知 $f(0.670)<0$,于是有

$$
0.670<\xi<0.671 \text {. }
$$

以 0.670 或 0.671 作为根的近似值, 其误差都小于 $10^{-3}$.

## 习 题 3-8

1. 试证明方程 $x^{3}-3 x^{2}+6 x-1=0$ 在区间 $(0,1)$ 内有唯一的实根, 并用二分法求这个 根的近似值, 使误差不超过 0.01 .
2. 试证明方程 $x^{5}+5 x+1=0$ 在区间 $(-1,0)$ 内有唯一的实根, 并用切线法求这个根的 近似值,使误差不超过 0.01 .
3. 求方程 $x^{3}+3 x-1=0$ 的近似根, 使误差不超过 0.01 .
4. 求方程 $x \lg x=1$ ，的近似根；使误差不超过 0.01 .

## 总习题三

## 1. 填空:

设常数 $k>0$, 函数 $f(x)=\ln x-\frac{x}{\mathrm{e}}+k$ 在 $(0,+\infty)$ 内零点的个数为

2. 选择以下两题中给出的四个结论中一个正确的结论:

(1) 设在 $[0,1]$ 上 $f^{\prime \prime}(x)>0$, 则 $f^{\prime}(0), f^{\prime}(1), f(1)-f(0)$ 或 $f(0)-f(1)$ 几个数的大小 欺序为 ( ):
(A) $f^{\prime}(1)>f^{\prime}(0)>f(1)-f(0)$.
(B) $f^{\prime}(1)>f(1)-f(0)>f^{\prime}(0)$.
(C) $f(1)-f^{\prime}(0)>f^{\prime \prime}(1)>f^{\prime}(0)$.
(D) $f^{\prime}(1)>f(0)-f(1)>f^{\prime}(0)$.

(2) 设 $f^{\prime}\left(x_{0}\right)=f^{\prime \prime}\left(x_{0}\right)=0, f^{\prime \prime \prime}\left(x_{0}\right)>0$, 则 () .

(A) $f^{\prime}\left(x_{0}\right)$ 是 $f^{\prime}(x)$ 的极大值. $\quad$ (B) $f\left(x_{0}\right)$ 是 $f(x)$ 的极大值.

(C) $f\left(x_{0}\right)$ 是 $f(x)$ 的极小值.

(D) $\left(x_{0}, f\left(x_{0}\right)\right)$ 是曲线 $y=f(x)$ 的拐点.

3. 列举一个函数 $f(x)$ 满足: $f(x)$ 在 $[a, b]$ 上连续, 在 $(a, b)$ 内除某一点外处处可导, 但 在 $(a, b)$ 内不存在点 $\xi$, 使 $f(b)-f(a)=f^{\prime}(\xi)(b-a)$.
4. 设 $\lim _{x \rightarrow \infty} f^{\prime}(x)=k$, 求 $\lim _{x \rightarrow \infty}[f(x+a)-f(x)]$.
5. 证明多项式 $f(x)=x^{3}-3 x+a$ 在 $[0,1]$ 上不可能有两个零点.
6. 设 $a_{0}+\frac{a_{1}}{2}+\cdots+\frac{a_{u}}{n+1}=0$, 证明多项式

$$
f(x)=a_{0}+a_{1} x+\cdots+a_{n} x^{n}
$$

在 $(0,1)$ 内至少有一个零点.

7. 设 $f(x)$ 在 $[0, a]$ 上连续, 在 $(0, a)$ 内可导, 且 $f(a)=0$, 证明存在一点 $\xi \in(0, a)$, 使 $f(\xi)+\xi f^{\circ}(\xi)=0$.

“8. 设 $0<a<b$, 函数 $f(x)$ 在 $[a, b]$ 上连续, 在 $(a, b)$ 内可导, 试利用柯西中值定理, 证明 存在一点 $\xi \in(a, b)$, 使

$$
f(b)-f(a)=\xi f^{\prime}(\xi) \ln \frac{b}{a} .
$$

9. 设 $f(x) 、 g(x)$ 都是可导函数, 且 $\left|f^{\prime}(x)\right|<g^{\prime}(x)$, 证明: 当 $x>a$ 时 $|f(x)-f(a)|$ $<g(x)-g(a)$.
10. 求下列极限:

(1) $\lim _{x \rightarrow 1} \frac{x-x^{x}}{1-x+\ln x}$;

(2) $\lim _{x \rightarrow 0}\left[\frac{1}{\ln (1+x)}-\frac{1}{x}\right]$;

(3) $\lim _{x \rightarrow+\infty}\left(\frac{2}{\pi} \arctan x\right)^{x}$;

(4) $\lim _{x \rightarrow \infty}\left[\left(a_{1}{ }^{\frac{1}{x}}+a_{2} \frac{1}{x}+\cdots+a_{n}^{\frac{1}{x}}\right) / n\right]^{n \cdot x}$ (其中 $a_{1}, a_{2}, \cdots, a_{n}>0$ ).

11. 证明下列不等式:

（1）当 $0<x_{1}<x_{2}<\frac{\pi}{2}$ 时, $\frac{\tan x_{2}}{\tan x_{1}}>\frac{x_{2}}{x_{1}}$;

（2）当 $x>0$ 时, $\ln (1+x)>\frac{\arctan x}{1+x}$;

（3）当 $\mathrm{e}<a<b<\mathrm{e}^{2}$ 时, $\ln ^{2} b-\ln ^{2} a>\frac{4}{\mathrm{e}^{2}}(b-a)$.

12. 设 $a>1, f(x)=a^{r}-a x$ 在 $(-\infty,+\infty)$ 内的驻点为 $x(a)$. 问 $a$ 为何值时, $x(a)$ 圾 小? 并求出最小值.
13. 求椭图 $x^{2}-x y+y^{2}=3$ 上纵坐标最大和最小的点.
14. 求数列 $\{\sqrt[n]{n}\}$ 的极大项.
15. 曲线弧 $y=\sin x(0<x<\pi)$ 上哪一点处的曲率半径最小? 求出该点处的曲率半径.
16. 证明方程 $x^{3}-5 x-2=0$ 只有一个正根, 并求此正根的近似值，精确到 $10^{-3}$.
17. 设 $f^{\prime \prime}\left(x_{0}\right)$ 存在, 证明

$$
\lim _{h \rightarrow 0} \frac{f\left(x_{0}+h\right)+f\left(x_{0}-h\right)-2 f\left(x_{0}\right)}{h^{2}}=f^{\prime \prime}\left(x_{0}\right) .
$$

18. 设 $f^{(n)}\left(x_{0}\right)$ 存在, 且 $f\left(x_{0}\right)=f^{\prime}\left(x_{0}\right)=\cdots=f^{(n)}\left(x_{0}\right)=0$, 证明

$$
f(x)=o\left[\left(x-x_{0}\right)^{n}\right]\left(x \rightarrow x_{0}\right) .
$$

19. 设 $f(x)$ 在 $(a, b)$ 内二阶可导, 且 $f^{\prime \prime}(x) \geqslant 0$. 证明对于 $(a, b)$ 内任意两点 $x_{1} 、 x_{2}$ 及 $0 \leqslant t \leqslant 1$, 有

$$
f\left[(1-t) x_{1}+t x_{2}\right] \leqslant(1-t \cdot) f\left(x_{1}\right)+t f\left(x_{2}\right) .
$$

20. 试确定常数 $a$ 和 $b$, 使 $f(x)=x-(a+b \cos x) \sin x$ 为当 $x \rightarrow 0$ 时关于 $x$ 的 5 阶无 穷小.

## 第四章 不定 积 分

在第二章中, 我们讨论了如何求一个函数的导函数问题, 本章将讨论它的反 问题, 即要寻求一个可导函数, 使它的导函数等于已知函数. 这是积分学的基本 问题之一.

## 第一节 不定积分的概念与性质

## 一、原函数与不定积分的概念

定义 1 如果在区间 $I$ 上, 可导函数 $F(x)$ 的导函数为 $f(x)$, 即对任一 $x \in I$, 都有

$$
F^{\prime}(x)=f(x) \text { 或 } \mathrm{d} F(x)=f(x) \mathrm{d} x,
$$

那么函数 $F(x)$ 就称为 $f(x)$ (或 $f(x) \mathrm{d} x$ ) 在区间 $I$ 上的原函数.

例如，因 $(\sin x)^{\prime}=\cos x$, 故 $\sin x$ 是 $\cos x$ 的一个原函数.

又如当 $x \in(1,+\infty)$ 时,

$$
\left[\ln \left(x+\sqrt{x^{2}-1}\right)\right]^{\prime}=\frac{1}{x+\sqrt{x^{2}-1}}\left(1+\frac{x}{\sqrt{x^{2}-1}}\right)=\frac{1}{\sqrt{x^{2}-1}},
$$

故 $\ln \left(x+\sqrt{x^{2}-1}\right)$ 是 $\frac{1}{\sqrt{x^{2}-1}}$ 在区间 $(1,+\infty)$ 内的原函数.

关于原函数,我们首先要问:一个函数具备什么条件, 能保证它的原函数一 定存在? 这个问题将在下一章中讨论, 这里先介绍一个结论.

原函数存在定理 如果函数 $f(x)$ 在区间 $I$ 上连续,那么在区间 $I$ 上存在 可导函数 $F(x)$, 使对任 $-x \in I$ 都有

$$
F^{\prime}(x)=f(x) \text {. }
$$

简单地说就是: 连续函数一定有原函数.

下面还要说明两点.

第一, 如果 $f(x)$ 在区间 $I$ 上有原函数, 即有一个函数 $F(x)$, 使对任一 $x \in I$, 都有 $F^{\prime}(x)=f(x)$, 那么, 对任何常数 $C$, 显然也有

$$
[F(x)+C]^{\prime}=f(x),
$$

即对任何常数 $C$, 函数 $F(x)+C$ 也是 $f(x)$ 的原函数. 这说明, 如果 $f(x)$ 有一 个原函数,那么 $f(x)$ 就有无限多个原函数. 第二, 如果在区间 $I$ 上 $F(x)$ 是 $f(x)$ 的一个原函数, 那么 $f(x)$ 的其他原函 数与 $F(x)$ 有什么关系?

设 $\Phi(x)$ 是 $f(x)$ 的另一个原函数, 即对任一 $x \in I$ 有

$$
\Phi^{\prime}(x)=f(x) \text {, }
$$

于是

$$
[\Phi(x)-F(x)]^{\prime}=\Phi^{\prime}(x)-F^{\prime}(x)=f(x)-f(x)=0 .
$$

在第三章第一节中已经知道, 在一个区间上导数恒为零的函数必为常数, 所以

$$
\Phi(x)-F(x)=C_{0} \quad \text { ( } C_{0} \text { 为某个常数). }
$$

这表明 $\Phi(x)$ 与 $F(x)$ 只差一个常数. 因此, 当 $C$ 为任意的常数时, 表达式

$$
F(x)+C
$$

就可表示 $f(x)$ 的任意一个原函数. 也就是说, $f(x)$ 的全体原函数所组成的集 合, 就是函数族.

$$
\{F(x)+C \mid-\infty<C<+\infty\} .
$$

由以上两点说明, 我们引进下述定义.

定义 2 在区间 $I$ 上, 函数 $f(x)$ 的带有任意常数项的原函数称为 $f(x)$ (或 $f(x) \mathrm{d} x$ ) 在区间 $I$ 上的不定积分, 记作

$$
\int f(x) \mathrm{d} x
$$

其中记号 $\int$ 称为积分号, $f(x)$ 称为被积函数, $f(x) \mathrm{d} x$ 称为被积表达式, $x$ 称为 积分变量.

由此定义及前面的说明可知, 如果 $F(x)$ 是 $f(x)$ 在区间 $I$ 上的一个原函 数, 那么 $F(x)+C$ 就是 $f(x)$ 的不定积分, 即

$$
\int f(x) \mathrm{d} x=F(x)+C .
$$

因而不定积分 $\int f(x) \mathrm{d} x$ 可以表示 $f(x)$ 的任意一个原函数.

例 1 求 $\int x^{2} \mathrm{~d} x$.

解 由于 $\left(\frac{x^{3}}{3}\right)^{\prime}=x^{2}$, 所以 $\frac{x^{3}}{3}$ 是 $x^{2}$ 的一个原函数. 因此

$$
\int x^{2} \mathrm{~d} x=\frac{x^{3}}{3}+C
$$

例 2 求 $\int \frac{1}{x} \mathrm{~d} x$.

解 当 $x>0$ 时, 由于 $(\ln x)^{\prime}=\frac{1}{x}$, 所以 $\ln x$ 是 $\frac{1}{x}$ 在 $(0,+\infty)$ 内的一个原函 数. 因此; 在 $(0,+\infty)$ 内,

$$
\int \frac{1}{x} \mathrm{~d} x=\ln x+C
$$

当 $x<0$ 时, 由于 $[\ln (-x)]^{\prime}=\frac{1}{-x}(-1)=\frac{1}{x}$, 所以 $\ln (-x)$ 是 $\frac{1}{x}$ 在 $(-\infty, 0)$ 内的一个原函数. 因此, 在 $(-\infty, 0)$ 内,

$$
\int \frac{1}{x} \mathrm{~d} x=\ln (-x)+C .
$$

把在 $x>0$ 及 $x<0$ 内的结果合起来, 可写作

$$
\int \frac{1}{x} \mathrm{~d} x=\ln |x|+C \text {. }
$$

例 3 设曲线通过点 $(1,2)$, 且其上任一点处的切线斜率等于这点横坐标的 两倍,求此曲线的方程.

解 设所求的曲线方程为 $y=f(x)$, 按题设, 曲线上任一点 $(x, y)$ 处的切 线斜率为

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=2 x
$$

即 $f(x)$ 是 $2 x$ 的一个原函数.

因为

$$
\int 2 x \mathrm{~d} x=x^{2}+C,
$$

故必有某个常数 $C$ 使 $f(x)=x^{2}+C$, 即曲线方程为 $y=x^{2}+C$. 因所求曲线通 过点 $(1,2)$, 故

$$
2=1+C, C=1
$$

于是所求曲线方程为

$$
y=x^{2}+1 .
$$

函数 $f(x)$ 的原函数的图形称为 $f(x)$ 的积分曲线. 本例即 是求函数 $2 x$ 的 通过点 $(1,2)$ 的那条积分曲线. 显然, 这条积分曲线可以由另一条积分曲线(例如 $y=x^{2}$ ) 经 $y$ 轴方向平移而得 (图 4-1).

例 4 质点以初速度 $v_{0}$ 铅直上扡, 不计阻力, 求它的运动规律.

解 所谓运动规律; 是指质点的位置关于时间 $t$ 的函数关系. 为表示质点的 位置, 取坐标系如下: 把质点所在的铅直线取作坐标轴, 指向朝上,轴与地面的交 点取作坐标原点. 设质点抛出时刻为 $t=0$, 当 $t=0$ 时质点所在位置的坐标为 $x_{0}$, 在时刻' $t$ 时坐标为 $x$ (图 4-2), $x=x(t$ ) 就是要求的函数.

按导数的物理意义知道,

$$
\frac{\mathrm{d} x}{\mathrm{~d} t}=v(t)
$$

即为质点在时刻 $t$ 时向上运动的速度 (如果 $v(t)<0$, 那么运动方向实际朝下).

又知

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=\frac{\mathrm{d} v}{\mathrm{~d} t}=a(t)
$$

即为质点在时刻 $t$ 时向上运动的加速度, 按题意, 有 $a(t)=-g$, 即

$$
\frac{\mathrm{d} v}{\mathrm{~d} t}=-g \text { 或 } \frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=-g .
$$

先求 $v(t)$. 由 $\frac{\mathrm{d} v}{\mathrm{~d} t}=-g$, 即 $v(t)$ 是 $(-g)$ 的原函数, 故

$$
v(t)=\int(-g) \mathrm{d} t=-g t+C_{1},
$$

由 $v(0)=v_{0}$, 得 $v_{0}=C_{1}$, 于是

$$
v(t)=-g t+v_{0} .
$$

再求 $x(t)$. 由 $\frac{\mathrm{d} x}{\mathrm{~d} t}=v(t)$, 即 $x(t)$ 是 $v(t)$ 的原函数, 故

$$
x(t)=\int v(t) \mathrm{d} t=\int\left(-g t+v_{0}\right) \mathrm{d} t=-\frac{1}{2} g t^{2}+v_{0} t+C_{2},
$$

由 $x(0)=x_{0}$, 得 $x_{0}=C_{2}$, 于是所求运动规律为

$$
x=-\frac{1}{2} g t^{2}+v_{0} t+x_{0}, t \in[0, T],
$$

其中 $T$ 表示质点落地的时刻.

从不定积分的定义,即可知下述关系:

由于 $\int f(x) \mathrm{d} x$ 是 $f(x)$ 的原函数, 所以

或

$$
\begin{aligned}
& \frac{\mathrm{d}}{\mathrm{d} x}\left[\int f(x) \mathrm{d} x\right]=f(x), \\
& \mathrm{d}\left[\int f(x) \mathrm{d} x\right]=f(x) \mathrm{d} x ;
\end{aligned}
$$

又由于 $F(x)$ 是 $F^{\prime}(x)$ 的原函数, 所以

$$
\int F^{\prime}(x) \mathrm{d} x=F(x)+C,
$$

或记作

$$
\int \mathrm{d} F(x)=F(x)+C .
$$

由此可见,微分运算 (以记号 $\mathrm{d}$ 表示) 与求不定积分的运算 (简称积分运算, 以记号 $\int$ 表示) 是互逆的. 当记号 $\int$ 与 $d$ 连在一起时, 或者抵消, 或者抵消后差一 个常数.

## 二、基本积分表

既然积分运算是微分运算的逆运算,那么很自然地可以从导数公式得到相 应的积分公式.

例如, 因为 $\left(\frac{x^{\mu+1}}{\mu+1}\right)^{\prime}=x^{\mu}$, 所以 $\frac{x^{\mu+1}}{\mu+1}$ 是 $x^{\mu}$ 的一个原函数, 于是

$$
\int x^{\mu} \mathrm{d} x=\frac{x^{\mu+1}}{\mu+1}+C \quad(\mu \neq-1),
$$

类似地可以得到其他积分公式.下面我们把一些基本的积分公式列成一个 表,这个表通常叫做基本积分表.

(1) $\int k \mathrm{~d} x=k x+C(k$ 是常数 $)$,

(2) $\int x^{\mu} \mathrm{d} x=\frac{x^{\mu+1}}{\mu+1}+C(\mu \neq-1)$,

(3) $\int \frac{\mathrm{d} x}{x}=\ln |x|+C$,

(4) $\int \frac{\mathrm{d} x}{1+x^{2}}=\arctan x+C$,

(5) $\int \frac{\mathrm{d} x}{\sqrt{1-x^{2}}}=\arcsin x+C$,

(6) $\int \cos x \mathrm{~d} x=\sin x+C$,

(7) $\int \sin x \mathrm{~d} x=-\cos x+C$,

(8) $\int \frac{\mathrm{d} x}{\cos ^{2} x}=\int \sec ^{2} x \mathrm{~d} x=\tan x+C$,

(9) $\int \frac{\mathrm{d} x}{\sin ^{2} x}=\int \csc ^{2} x \mathrm{~d} x=-\cot x+C$,

(111) $\int \sec x \tan x \mathrm{~d} x=\sec x+C$, (11) $\int \csc x \cot x \mathrm{~d} x=-\csc x+C$, .

(12) $\int \mathrm{e}^{x} \mathrm{~d} x=\mathrm{e}^{x}+C$,

(13) $\int a^{x} \mathrm{~d} x=\frac{a^{x}}{\ln a}+C$.

以上十三个基本积分公式是求不定积分的基础, 必须熟记,下面举几个应用 筆函数的积分公式(2)的例子.

例 5 求 $\int \frac{\mathrm{d} x}{x^{3}}$.

解 $\int \frac{\mathrm{d} x}{x^{3}}=\int x^{-3} \mathrm{~d} x=\frac{x^{-3+1}}{-3+1}+C=-\frac{1}{2 x^{2}}+C$ ．

例 6 求 $\int x^{2} \sqrt{x} \mathrm{~d} x$.

解 $\int x^{2} \sqrt{x} \mathrm{~d} x=\int x^{\frac{5}{2}} \mathrm{~d} x=\frac{x^{\frac{5}{2}+1}}{\frac{5}{2}+1}+C=\frac{2}{7} x^{\frac{7}{2}}+C$

$$
=\frac{2}{7} x^{3} \sqrt{x}+C \text {. }
$$

例 7 求 $\int \frac{\mathrm{d} x}{x \sqrt[3]{x}}$.

解 $\int \frac{\mathrm{d} x}{x \sqrt[3]{x}}=\int x^{-\frac{4}{3}} \mathrm{~d} x=\frac{x^{-\frac{4}{3}+1}}{-\frac{4}{3}+1}+C=-3 x^{-\frac{1}{3}}+C$

$$
=-\frac{3}{\sqrt[3]{x}}+C
$$

上面三个例子表明,有时被积函数实际是函数, 但用分式或根式表示. 遇 此情形,应先把它化为 $x^{\mu}$ 的形式, 然后应用幂函数的积分公式 (2)来求不定 积分.

## 三、不定积分的性质

根据不定积分的定义,可以推得它有如下两个性质：

性质 1 设函数 $f(x)$ 及 $g(x)$ 的原函数存在, 则

$$
\int[f(x)+g(x)] \mathrm{d} x=\int f(x) \mathrm{d} x+\int g(x) \mathrm{d} x .
$$

证 将(3)式右端求导, 得

$$
\begin{aligned}
{\left[\int f(x) \mathrm{d} x+\int g(x) \mathrm{d} x\right]^{\prime} } & =\left[\int f(x) \mathrm{d} x\right]^{\prime}+\left[\int g(x) \mathrm{d} x\right]^{\prime} \\
& =f(x)+g(x) .
\end{aligned}
$$

这表示,(3)式右端是 $f(x)+g(x)$ 的原函数, 又(3) 式右端有两个积分记号,形 式上含两个任意常数, 由于任意常数之和仍为任意常数, 故实际上含一个任意常 数,因此 (3) 式右端是 $f(x)+g(x)$ 的不定积分.

性质 1 对于有限个函数都是成立的.

类似地可以证明不定积分的第二个性质.

性质 2 设函数 $f(x)$ 的原函数存在, $k$ 为非零常数, 则

$$
\int k f(x) \mathrm{d} x=k \int f(x) \mathrm{d} x .
$$

利用基本积分表以及不定积分的这两个性质, 可以求出一些简单函数的不 定积分.

例 8 求 $\int \sqrt{x}\left(x^{2}-5\right) \mathrm{d} x$.

$$
\text { 解 } \begin{aligned}
\int \sqrt{x}\left(x^{2}-5\right) \mathrm{d} x & =\int\left(x^{\frac{5}{2}}-5 x^{\frac{1}{2}}\right) \mathrm{d} x \\
& =\int x^{\frac{5}{2}} \mathrm{~d} x-\int 5 x^{\frac{1}{2}} \mathrm{~d} x \\
& =\int x^{\frac{5}{2}} \mathrm{~d} x-5 \int x^{\frac{1}{2}} \mathrm{~d} x \\
& =\frac{2}{7} x^{\frac{7}{2}}-5 \cdot \frac{2}{3} x^{\frac{3}{2}}+C \\
& =\frac{2}{7} x^{3} \sqrt{x}-\frac{10}{3} x \sqrt{x}+C .
\end{aligned}
$$

注意 检验积分结果是否正确, 只要对结果求导, 看它的导数是否等于被积 函数,相等时结果是正确的,否则结果是错误的. 如就例 8 的结果来看, 由于

所以结果是正确的.

$$
\begin{aligned}
\left(\frac{2}{7} x^{3} \sqrt{x}-\frac{10}{3} x \sqrt{x}+C\right)^{\prime} & =\left(\frac{2}{7} x^{\frac{7}{2}}-\frac{10}{3} x^{\frac{3}{2}}+C\right)^{\prime} \\
& =x^{\frac{5}{2}}-5 x^{\frac{1}{2}}=\sqrt{x}\left(x^{2}-5\right),
\end{aligned}
$$

例 9 求 $\int \frac{(x-1)^{3}}{x^{2}} \mathrm{~d} x$.

解 $\int \frac{(x-1)^{3}}{x^{2}} \mathrm{~d} x=\int \frac{x^{3}-3 x^{2}+3 x-1}{x^{2}} \mathrm{~d} x$

$$
\begin{aligned}
& =\int\left(x-3+\frac{3}{x}-\frac{1}{x^{2}}\right) \mathrm{d} x \\
& =\int x \mathrm{~d} x-3 \int \mathrm{d} x+3 \int \frac{\mathrm{d} x}{x}-\int \frac{\mathrm{d} x}{x^{2}}
\end{aligned}
$$

$$
=\frac{x^{2}}{2}-3 x+3 \ln |x|+\frac{1}{x}+C .
$$

例 10 求 $\int\left(\mathrm{e}^{x}-3 \cos x\right) \mathrm{d} x$.

解 $\int\left(\mathrm{e}^{\mathrm{x}}-3 \cos x\right) \mathrm{d} x=\int \mathrm{e}^{\mathrm{x}} \mathrm{d} x-3 \int \cos x \mathrm{~d} x$

$$
=\mathrm{e}^{x}-3 \sin x+C \text {. }
$$

例 11 求 $\int 2^{x} \mathrm{e}^{x} \mathrm{~d} x$.

解 因为

$$
2^{x} \mathrm{e}^{x}=(2 \mathrm{e})^{x},
$$

所以可把 $2 \mathrm{e}$ 看作 $a$,并利用积分公式 (3), 便得

$$
\int 2^{x} \mathrm{e}^{x} \mathrm{~d} x=\int(2 \mathrm{e})^{x} \mathrm{~d} x=\frac{(2 \mathrm{e})^{x}}{\ln (2 \mathrm{e})}+C=\frac{2^{x} \mathrm{e}^{x}}{1+\ln 2}+C .
$$

例 12 求 $\int \tan ^{2} x \mathrm{~d} x$.

解 基本积分表中没有这种类型的积分,先利用三角恒等式化成表中所列 类型的积分, 然后再逐项求积分:

$$
\begin{aligned}
\int \tan ^{2} x \mathrm{~d} x & =\int\left(\sec ^{2} x-1\right) \mathrm{d} x=\int \sec ^{2} x \mathrm{~d} x-\int \mathrm{d} x \\
& =\tan x-x+C .
\end{aligned}
$$

例 13 求 $\int \sin ^{2} \frac{x}{2} \mathrm{~d} x$.

解 基本积分表中也没有这种类型的积分, 同上例一样, 可以先利用三角恒 等式变形,然后再逐项求积分:

$$
\begin{aligned}
\int \sin ^{2} \frac{x}{2} \mathrm{~d} x & =\int \frac{1}{2}(1-\cos x) \mathrm{d} x=\frac{1}{2} \int(1-\cos x) \mathrm{d} x \\
& =\frac{1}{2}\left(\int \mathrm{d} x-\int \cos x \mathrm{~d} x\right)=\frac{1}{2}(x-\sin x)+C .
\end{aligned}
$$

例 14 求 $\int \frac{1}{\sin ^{2} \frac{x}{2} \cos ^{2} \frac{x}{2}} \mathrm{~d} x$.

解 同上例一样,先利用三角恒等式变形,然后再求积分:

$$
\begin{aligned}
\int \frac{1}{\sin ^{2} \frac{x}{2} \cos ^{2} \frac{x}{2}} \mathrm{~d} x & =\int \frac{1}{\left(\frac{\sin x}{2}\right)^{2}} \mathrm{~d} x \\
& =4 \int \csc ^{2} x \mathrm{~d} x=-4 \cot x+C .
\end{aligned}
$$

例 15 求 $\int \frac{2 x^{4}+x^{2}+3}{x^{2}+1} \mathrm{~d} x$. 解 被积函数的分子和分母都是多项式, 通过多项式的除法, 可以把它化成 基本积分表中所列类型的积分,然后再逐项求积分:

$$
\begin{aligned}
\int \frac{2 x^{4}+x^{2}+3}{x^{2}+1} \mathrm{~d} x & =\int\left(2 x^{2}-1+\frac{4}{x^{2}+1}\right) \mathrm{d} x \\
& =2 \int x^{2} \mathrm{~d} x-\int 1 \mathrm{~d} x+4 \int \frac{1}{x^{2}+1} \mathrm{~d} x \\
& =\frac{2}{3} x^{3}-x+4 \arctan x+C .
\end{aligned}
$$

## 习 题 4-1

1. 利用求导运算验证下列等式:

(1) $\int \frac{1}{\sqrt{x^{2}+1}} \mathrm{~d} x=\ln \left(x+\sqrt{x^{2}+1}\right)+C$;

(2) $\int \frac{1}{x^{2} \sqrt{x^{2}-1}} \mathrm{~d} x=\frac{\sqrt{x^{2}-1}}{x}+C$;

(3) $\int \frac{2 x}{\left(x^{2}+1\right)(x+1)^{2}} \mathrm{~d} x=\arctan x+\frac{1}{x+1}+C$;

(4) $\int \sec x \mathrm{~d} x=\ln |\tan x+\sec x|+C$;

(5) $\int x \cos x \mathrm{~d} x=x \sin x+\cos x+C$;

(6) $\int \mathrm{e}^{x} \sin x \mathrm{~d} x=\frac{1}{2} \mathrm{e}^{x}(\sin x-\cos x)+C$.

2. 求下列不定积分:
(1) $\int \frac{\mathrm{d} x}{x^{2}}$;
(2) $\int x \sqrt{x} \mathrm{~d} x$;
(3) $\int \frac{\mathrm{d} x}{\sqrt{x}}$;
(4) $\int x^{2} \sqrt[3]{x} \mathrm{~d} x$;
(5) $\int \frac{\mathrm{d} x}{x^{2} \sqrt{x}}$;
(6) $\int \sqrt[m]{x^{n}} \mathrm{~d} x$;
(7) $\int 5 x^{3} \mathrm{~d} x$;
(8) $\int\left(x^{2}-3 x+2\right) \mathrm{d} x$;
(9) $\int \frac{\mathrm{d} h}{\sqrt{2 g h}}(g$ 是常数);
(10) $\int\left(x^{2}+1\right)^{2} \mathrm{~d} x$;
(11) $\int(\sqrt{x}+1)\left(\sqrt{x^{3}}-1\right) \mathrm{d} x$;
(12) $\int \frac{(1-x)^{2}}{\sqrt{x}} \mathrm{~d} x$;
(13) $\int\left(2 \mathrm{e}^{x}+\frac{3}{x}\right) \mathrm{d} x$;
(14) $\int\left(\frac{3}{1+x^{2}}-\frac{2}{\sqrt{1-x^{2}}}\right) \mathrm{d} x$;
(15) $\int \mathrm{e}^{x}\left(1-\frac{\mathrm{e}^{-x}}{\sqrt{x}}\right) \mathrm{d} x$;
(16) $\int 3^{x} \mathrm{e}^{x} \mathrm{~d} x$;
(17) $\int \frac{2 \cdot 3^{r}-5 \cdot 2^{r}}{3^{r}} \mathrm{~d} x$;
(18) $\int \sec x(\sec x-\tan x) \mathrm{d} x$;
(19) $\int \cos ^{2} \frac{x}{2} \mathrm{~d} x$;
(20) $\int \frac{\mathrm{d} x}{1+\cos 2 x}$;
(21) $\int \frac{\cos 2 x}{\cos x-\sin x} \mathrm{~d} x$;
(22) $\int \frac{\cos 2 x}{\cos ^{2} x \sin ^{2} x} \mathrm{~d} x$;
(23) $\int \cot ^{2} x \mathrm{~d} x$;
(24) $\int \cos \theta(\tan \theta+\sec \theta) \mathrm{d} \theta$;
(25) $\int \frac{x^{2}}{x^{2}+1} \mathrm{~d} x$;
(26) $\int \frac{3 x^{4}+2 x^{2}}{x^{2}+1} \mathrm{~d} x$.
3. 含有未知函数的导数的方程称为做分方程，例如方程 $\frac{\mathrm{d} y}{\mathrm{~d} x}=f(x)$,其中 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ 为未知函数 的导数, $f(x)$ 为已知函数. 如果将函数 $y=\varphi(x)$ 代入微分方程, 使微分方程成为恒等式, 那么 函数 $y=\varphi(x)$ 就称为这微分方程的解. 求下列微分方程满足所给条件的解:
(1) $\frac{\mathrm{d} y}{\mathrm{~d} x}=(x-2)^{2},\left.\quad y\right|_{x=2}=0$;
(2) $\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=\frac{2}{t^{3}},\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=1}=1,\left.x\right|_{1=1}=1$.
4. 汽车以 $20 \mathrm{~m} / \mathrm{s}$ 的速度行驶, 刹车后匀减速行驶了 $50 \mathrm{~m}$ 停住, 求刹车加速度. 可执行下 列步槡:

（1）求微分方程 $\frac{\mathrm{d}^{2} s}{\mathrm{~d} t^{2}}=-k$ 满足条件 $\left.\frac{\mathrm{d} s}{\mathrm{~d} t}\right|_{t=0}=20$ 及 $\left.s\right|_{t=0}=0$ 的解；

(2) 求使 $\frac{\mathrm{d} s}{\mathrm{~d} t}=0$ 的 $t$ 值及相应的 $s$ 值;

(3) 求使 $s=50$ 的 $k$ 值.

5. 一曲线通过点 $\left(\mathrm{e}^{2}, 3\right)$, 且在任一点处的切线的斜率等于该点栱坐标的倒数, 求该曲线 的方程.
6. 一物体由静止开始运动，经 $t(\mathrm{~s})$ 后的速度是 $3 t^{2}(\mathrm{~m} / \mathrm{s})$, 问

(1) 在 $3 \mathrm{~s}$ 后物体离开出发点的距离是多少?

（2）物体走完 $360 \mathrm{~m}$ 需要多少时间?

7. 证明函数 $\arcsin (2 x-1), \arccos (1-2 x)$ 和 $2 \arctan \sqrt{\frac{x}{1-x}}$ 都是 $\frac{1}{\sqrt{x-x^{2}}}$ 的原函数.

## 第二节 换元积分法

利用基本积分表与积分的性质, 所能计算的不定积分是非常有限的. 因此, 有必要进一步来研究不定积分的求法. 本节把复合函数的微分法反过来用于求 不定积分, 利用中间变量的代换, 得到复合函数的积分法, 称为热元积分法, 简称 热元法.换元法通常分成两类,下面先讲第一类换元法.

## 一、第一类换元法

设 $f(u)$ 具有原函数 $F(u)$, 即

$$
F^{\prime}(u)=f(u), \int f(u) \mathrm{d} u=F(u)+C .
$$

如果 $u$ 是中间变量: $u=\varphi(x)$, 且设 $\varphi(x)$ 可微, 那么, 根据复合函数微分法, 有

$$
\mathrm{d} F[\varphi(x)]=f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x,
$$

从而根据不定积分的定义就得

$$
\int f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x=F[\varphi(x)]+C=\left[\int f(u) \mathrm{d} u\right]_{u=\varphi(x)} .
$$

于是有下述定理：

定理 1 设 $f(u)$ 具有原函数, $u=\varphi(x)$ 可导, 则有换元公式

$$
\int f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x=\left[\int f(u) \mathrm{d} u\right]_{n=\varphi(x)} .
$$

由此定理可见,虽然 $\int f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x$ 是一个整体的记号, 但从形式上 看,被积表达式中的 $\mathrm{d} x$ 也可当作变量 $x$ 的微分来对待, 从而微分等式 $\varphi^{\prime}(x) \mathrm{d} x$ $=\mathrm{d} u$ 可以方便地应用到被积表达式中来,我们在上节第一目中已经这样用了, 那里把积分 $\int F^{\prime}(x) \mathrm{d} x$ 记作 $\int \mathrm{d} F(x)$, 就是按微分 $F^{\prime}(x) \mathrm{d} x=\mathrm{d} F(x)$, 把被积 表达式 $F^{\prime}(x) \mathrm{d} x$ 记作 $\mathrm{d} F(x)$.

如何应用公式 (1) 来求不定积分? 设要求 $\int g(x) \mathrm{d} x$, 如果函数 $g(x)$ 可以 化为 $g(x)=f[\varphi(x)] \varphi^{\prime}(x)$ 的形式, 那么

$$
\int g(x) \mathrm{d} x=\int f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x=\left[\int f(u) \mathrm{d} u\right]_{u=\varphi(x)}, .
$$

这样, 函数 $g(x)$ 的积分即转化为函数 $f(u)$ 的积分. 如果能求得 $f(u)$ 的原函数, 那么也就得到了 $g(x)$ 的原函数.

例 1 求 $\int 2 \cos 2 x \mathrm{~d} x$.

解 被积函数中, $\cos 2 x$ 是一个复合函数: $\cos 2 x=\cos u, u=2 x$, 常数因子 恰好是中间变量 $u$ 的导数. 因此,作变换 $u=2 x$, 便有

$$
\begin{aligned}
\int 2 \cos 2 x \mathrm{~d} x & =\int \cos 2 x \cdot 2 \mathrm{~d} x=\int \cos 2 x \cdot(2 x)^{\prime} \mathrm{d} x \\
& =\int \cos u \mathrm{~d} u=\sin u+C,
\end{aligned}
$$

再以 $u=2 x$ 代入, 即得

$$
\int 2 \cos 2 x \mathrm{~d} x=\sin 2 x+C .
$$

例 2 求 $\int \frac{1}{3+2 x} \mathrm{~d} x$.

解 被积函数 $\frac{1}{3+2 x}=\frac{1}{u}, u=3+2 x$. 这里缺少 $\frac{\mathrm{d} u}{\mathrm{~d} x}=2$ 这样一个因子, 但由 于 $\frac{\mathrm{d} u}{\mathrm{~d} x}$ 是个常数,故可改变系数凑出这个因子:

$$
\frac{1}{3+2 x}=\frac{1}{2} \cdot \frac{1}{3+2 x} \cdot 2=\frac{1}{2} \cdot \frac{1}{3+2 x}(3+2 x)^{\prime},
$$

从而令 $u=3+2 x$, 便有

$$
\begin{aligned}
\int \frac{1}{3+2 x} \mathrm{~d} x & =\int \frac{1}{2} \cdot \frac{1}{3+2 x}(3+2 x)^{\prime} \mathrm{d} x=\int \frac{1}{2} \cdot \frac{1}{u} \mathrm{~d} u \\
& =\frac{1}{2} \ln |u|+C=\frac{1}{2} \ln |3+2 x|+C .
\end{aligned}
$$

一般的, 对于积分 $\int f(a x+b) \mathrm{d} x$, 总可作变换 $u=a x+b$, 把它化为

$$
\begin{aligned}
\int f(a x+b) \mathrm{d} x & =\int \frac{1}{a} f(a x+b) \mathrm{d}(a x+b) \\
& =\frac{1}{a}\left[\int f(u) \mathrm{d} u\right]_{u=a r+b} .
\end{aligned}
$$

例 3 求 $\int \frac{x^{2}}{(x+2)^{3}} \mathrm{~d} x$.

解 令 $u=x+2$, 则 $x=u-2, \mathrm{~d} x=\mathrm{d} u$. 于是

$$
\begin{aligned}
\int \frac{x^{2}}{(x+2)^{3}} \mathrm{~d} x & =\int \frac{(u-2)^{2}}{u^{3}} \mathrm{~d} u=\int\left(u^{2}-4 u+4\right) u^{-3} \mathrm{du} \cdot \\
& =\int\left(u^{-1}-4 u^{-2}+4 u^{-3}\right) \mathrm{d} u \\
& =\ln |u|+4 u^{-1}-2 u^{-2}+C \\
& =\ln |x+2|+\frac{4}{x+2}-\frac{2}{(x+2)^{2}}+C .
\end{aligned}
$$

例 4 求 $\int 2 x \mathrm{e}^{x^{2}} \mathrm{~d} x$.

解 被积函数中的一个因子为 $\mathrm{e}^{x^{2}}=\mathrm{e}^{u}, u=x^{2}$; 剩下的因子 $2 x$ 恰好是中间 变量 $u=x^{2}$ 的导数, 于是有

$$
\int 2 x \mathrm{e}^{x^{2}} \mathrm{~d} x=\int \mathrm{e}^{x^{2}} \mathrm{~d}\left(x^{2}\right)=\int \mathrm{e}^{u} \mathrm{~d} u=\mathrm{e}^{u}+C=\mathrm{e}^{x^{2}}+C .
$$

例 5 求 $\int x \sqrt{1-x^{2}} \mathrm{~d} x$. 解 设 $u=1-x^{2}$, 则 $\mathrm{d} u=-2 x \mathrm{~d} x$, 即 $-\frac{1}{2} \mathrm{~d} u=x \mathrm{~d} x$, 因此,

$$
\begin{aligned}
\int x \sqrt{1-x^{2}} \mathrm{~d} x & =\int u^{\frac{1}{2}} \cdot\left(-\frac{1}{2}\right) \mathrm{d} u=-\frac{1}{2} \frac{u^{\frac{3}{2}}}{\frac{3}{2}}+C \\
& =-\frac{1}{3} u^{\frac{3}{2}}+C=-\frac{1}{3}\left(1-x^{2}\right)^{\frac{3}{2}}+C .
\end{aligned}
$$

在对变量代换比较熟练以后, 就不一定写出中间变量 $u$.

例 6 求 $\int \frac{1}{a^{2}+x^{2}} \mathrm{~d} x$.

解 $\int \frac{1}{a^{2}+x^{2}} \mathrm{~d} x=\int \frac{1}{a^{2}} \cdot \frac{1}{1+\left(\frac{x}{a}\right)^{2}} \mathrm{~d} x$

$$
=\frac{1}{a} \int \frac{1}{1+\left(\frac{x}{a}\right)^{2}} \mathrm{~d} \frac{x}{a}=\frac{1}{a} \arctan \frac{x}{a}+C .
$$

在上例中,我们实际上已经用了变量代换 $u=\frac{x}{a}$, 并在求出积分 $\frac{1}{a} \int \frac{1}{1+u^{2}} \mathrm{~d} u$ 之后,代回了原积分变量 $x$,只是没有把这些步骤写出来而已.

例 7 求 $\int \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}} \quad(a>0)$.

解 $\int \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}}=\int \frac{1}{a} \frac{\mathrm{d} x}{\sqrt{1-\left(\frac{x}{a}\right)^{2}}}=\int \frac{\mathrm{d} \frac{x}{a}}{\sqrt{1-\left(\frac{x}{a}\right)^{2}}}$

$$
=\arcsin \frac{x}{a}+C \text {. }
$$

例 8 求 $\int \frac{1}{x^{2}-a^{2}} \mathrm{~d} x$.

解 由于

$$
\frac{1}{x^{2}-a^{2}}=\frac{1}{2 a}\left(\frac{1}{x-a}-\frac{1}{x+a}\right),
$$

所以

$$
\begin{aligned}
\int \frac{1}{x^{2}-a^{2}} \mathrm{~d} x & =\frac{1}{2 a} \int\left(\frac{1}{x-a}-\frac{1}{x+a}\right) \mathrm{d} x \\
& =\frac{1}{2 a}\left(\int \frac{1}{x-a} \mathrm{~d} x-\int \frac{1}{x+a} \mathrm{~d} x\right) \\
& =\frac{1}{2 a}\left[\int \frac{1}{x-a} \mathrm{~d}(x-a)-\int \frac{1}{x+a} \mathrm{~d}(x+a)\right]
\end{aligned}
$$

$$
\begin{aligned}
& =\frac{1}{2 a}(\ln |x-a|-\ln |x+a|)+C \\
& =\frac{1}{2 a} \ln \left|\frac{x-a}{x+a}\right|+C
\end{aligned}
$$

例 9 求 $\int \frac{\mathrm{d} x}{x(1+2 \ln x)}$.

解 $\int \frac{\mathrm{d} x}{x(1+2 \ln x)}=\int \frac{\mathrm{d}(\ln x)}{1+2 \ln x}$

$$
=\frac{1}{2} \int \frac{\mathrm{d}(1+2 \ln x)}{1+2 \ln x}=\frac{1}{2} \ln |1+2 \ln x|+C .
$$

例 10 求 $\int \frac{\mathrm{e}^{3 \sqrt{x}}}{\sqrt{x}} \mathrm{~d} x$.

解 由于 $\mathrm{d} \sqrt{x}=\frac{1}{2} \frac{\mathrm{d} x}{\sqrt{x}}$, 因此,

$$
\begin{aligned}
\int \frac{\mathrm{e}^{3 \sqrt{x}}}{\sqrt{x}} \mathrm{~d} x & =2 \int \mathrm{e}^{3 \sqrt[3]{x}} \mathrm{~d} \sqrt{x}=\frac{2}{3} \int \mathrm{e}^{3 \sqrt{x}} \mathrm{~d}(3 \sqrt{x}) \\
& =\frac{2}{3} \mathrm{e}^{3 \sqrt{x}}+C .
\end{aligned}
$$

下面再举一些积分的例子,它们的被积函数中含有三角函数,在计算这种积 分的过程中, 往往要用到一些三角恒等式.

例 11 求 $\int \sin ^{3} x \mathrm{~d} x$.

解 $\int \sin ^{3} x \mathrm{~d} x=\int \sin ^{2} x \sin x \mathrm{~d} x=-\int\left(1-\cos ^{2} x\right) \mathrm{d}(\cos x)$

$$
=-\cos x+\frac{1}{3} \cos ^{3} x+C \text {. }
$$

例 12 求 $\int \sin ^{2} x \cos ^{5} x \mathrm{~d} x$.

解

$$
\begin{aligned}
\int \sin ^{2} x \cos ^{5} x \mathrm{~d} x & =\int \sin ^{2} x \cos ^{4} x \cos x \mathrm{~d} x \\
& =\int \sin ^{2} x\left(1-\sin ^{2} x\right)^{2} \mathrm{~d}(\sin x) \\
& =\int\left(\sin ^{2} x-2 \sin ^{4} x+\sin ^{6} x\right) \mathrm{d}(\sin x) \\
& =\frac{1}{3} \sin ^{3} x-\frac{2}{5} \sin ^{5} x+\frac{1}{7} \sin ^{7} x+C .
\end{aligned}
$$

一般的, 对于 $\sin ^{2 k+1} x \cos ^{n} x$ 或 $\sin ^{n} x \cos ^{2 k+1} x$ (其中 $k \in \mathrm{N}$ ) 型函数的积分, 总可依次作变换 $u=\cos x$ 或 $u=\sin x$, 求得结果.

例 13 求 $\int \tan x \mathrm{~d} x$. 解

$$
\begin{aligned}
\int \tan x \mathrm{~d} x & =\int \frac{\sin x}{\cos x} \mathrm{~d} x=-\int \frac{1}{\cos x} \mathrm{~d}(\cos x) \\
& =-\ln |\cos x|+C_{0}
\end{aligned}
$$

类似地可得

$$
\int \cot x \mathrm{~d} x=\ln |\sin x|+C \text {. }
$$

例 14 求 $\int \cos ^{2} x \mathrm{~d} x$.

$$
\text { 解 } \begin{aligned}
\int \cos ^{2} x \mathrm{~d} x & =\int \frac{1+\cos 2 x}{2} \mathrm{~d} x=\frac{1}{2}\left(\int \mathrm{d} x+\int \cos 2 x \mathrm{~d} x\right) \\
& =\frac{1}{2} \int \mathrm{d} x+\frac{1}{4} \int \cos 2 x \mathrm{~d}(2 x) \\
& =\frac{x}{2}+\frac{\sin 2 x}{4}+C .
\end{aligned}
$$

例 15 求 $\int \sin ^{2} x \cos ^{4} x \mathrm{~d} x$

解 $\int \sin ^{2} x \cos ^{4} x \mathrm{~d} x=\frac{1}{8} \int(1-\cos 2 x)(1+\cos 2 x)^{2} \mathrm{~d} x$

$$
\begin{aligned}
& =\frac{1}{8} \int\left(1+\cos 2 x-\cos ^{2} 2 x-\cos ^{3} 2 x\right) \mathrm{d} x \\
& =\frac{1}{8} \int\left(\cos 2 x-\cos ^{3} 2 x\right) \mathrm{d} x+\frac{1}{8} \int\left(1-\cos ^{2} 2 x\right) \mathrm{d} x \\
& =\frac{1}{8} \int \sin ^{2} 2 x \cdot \frac{1}{2} \mathrm{~d}(\sin 2 x)+\frac{1}{8} \int \frac{1}{2}(1-\cos 4 x) \mathrm{d} x \\
& =\frac{1}{48} \sin ^{3} 2 x+\frac{x}{16}-\frac{1}{64} \sin 4 x+C .
\end{aligned}
$$

一般的,对于 $\sin ^{2 k} x \cos ^{2 t} x(k 、 l \in \mathrm{N})$ 型函数, 总可利用三角恒等式: $\sin ^{2} x=$ $\frac{1}{2}(1-\cos 2 x), \cos ^{2} x=\frac{1}{2}(1+\cos 2 x)$ 化成 $\cos 2 x$ 的多项式, 然后采用例 15 中 所用的方法求得积分的结果.

例 16 求 $\int \sec ^{6} x \mathrm{~d} x$.

$$
\text { 解 } \begin{aligned}
\int \sec ^{6} x \mathrm{~d} x & =\int\left(\sec ^{2} x\right)^{2} \sec ^{2} x \mathrm{~d} x \\
& =\int\left(1+\tan ^{2} x\right)^{2} \mathrm{~d}(\tan x) \\
& =\int\left(1+2 \tan ^{2} x+\tan ^{4} x\right) \mathrm{d}(\tan x) \\
& =\tan x+\frac{2}{3} \tan ^{3} x+\frac{1}{5} \tan ^{5} x+C .
\end{aligned}
$$

例 17 求 $\int \tan ^{5} x \sec ^{3} x \mathrm{~d} x$. 解 $\int \tan ^{5} x \sec ^{3} x \mathrm{~d} x=\int \tan ^{4} x \sec ^{2} x \sec x \tan x \mathrm{~d} x$

$$
\begin{aligned}
& =\int\left(\sec ^{2} x-1\right)^{2} \sec ^{2} x \mathrm{~d}(\sec x) \\
& =\int\left(\sec ^{6} x-2 \sec ^{4} x+\sec ^{2} x\right) \mathrm{d}(\sec x) \\
& =\frac{1}{7} \sec ^{7} x-\frac{2}{5} \sec ^{5} x+\frac{1}{3} \sec ^{3} x+C .
\end{aligned}
$$

一般的, 对于 $\tan ^{n} x \sec ^{2 k} x$ 或 $\tan ^{2 k-1} x \sec ^{n} x\left(k \in \mathbf{N}^{+}\right)$型函数的积分, 可依次 作变换 $u=\tan x$ 或 $u=\sec x$, 求得结果.

例 18 求 $\int \csc x \mathrm{~d} x$.

、解 $\int \csc x \mathrm{~d} x=\int \frac{\mathrm{d} x}{\sin x}=\int \frac{\mathrm{d} x}{2 \sin \frac{x}{2} \cos \frac{x}{2}}$

$$
\begin{aligned}
& =\int \frac{\mathrm{d} \frac{x}{2}}{\tan \frac{x}{2} \cos ^{2} \frac{x}{2}}=\int \frac{\mathrm{d}\left(\tan \frac{x}{2}\right)}{\tan \frac{x}{2}} \\
& =\ln \left|\tan \frac{x}{2}\right|+C .
\end{aligned}
$$

因为

$$
\tan \frac{x}{2}=\frac{\sin \frac{x}{2}}{\cos \frac{x}{2}}=\frac{2 \sin ^{2} \frac{x}{2}}{\sin x}=\frac{1-\cos x}{\sin x}=\csc x-\cot x,
$$

所以上述不定积分又可表为:

$$
\int \csc x \mathrm{~d} x=\ln |\csc x-\cot x|+C .
$$

例 19 求 $\int \sec x \mathrm{~d} x$.

解 利用上例的结果, 有

$$
\begin{aligned}
\int \sec x \mathrm{~d} x & =\int \csc \left(x+\frac{\pi}{2}\right) \mathrm{d}\left(x+\frac{\pi}{2}\right) \\
& =\ln \left|\csc \left(x+\frac{\pi}{2}\right)-\cot \left(x+\frac{\pi}{2}\right)\right|+C \\
& =\ln |\sec x+\tan x|+C .
\end{aligned}
$$

例 20 求 $\int \cos 3 x \cos 2 x \mathrm{~d} x$.

解 利用三角函数的积化和差公式

$$
\cos A \cos B=\frac{1}{2}[\cos (A-B)+\cos (A+B)]
$$

得

$$
\cos 3 x \cos 2 x=\frac{1}{2}(\cos x+\cos 5 x),
$$

于是

$$
\begin{aligned}
\int \cos 3 x \cos 2 x \mathrm{~d} x & =\frac{1}{2} \int(\cos x+\cos 5 x) \mathrm{d} x \\
& =\frac{1}{2}\left(\int \cos x \mathrm{~d} x+\frac{1}{5} \int \cos 5 x \mathrm{~d}(5 x)\right) \\
& =\frac{1}{2} \sin x+\frac{1}{10} \sin 5 x+C .
\end{aligned}
$$

上面所举的例子, 可以使我们认识到公式 (1) 在求不定积分中所起的作用. 像复合函数的求导法则在微分学中一样, 公式 (1) 在积分学中也是经常使用的. 但利用公式 (1) 来求不定积分,一般却比利用复合函数的求导法则求函数的导数 要来得困难, 因为其中需要一定的技巧, 而且如何适当地选择变量代换 $u=$ $\varphi(x)$ 没有一般规律可循, 因此要掌握换元法, 除了熟悉一些典型的例子外, 还要 做较多的练习才行.

上述各例用的都是第一类换元法,即形如 $u=\varphi(x)$ 的变量代换.下面介绍 另一种形式的变量代换 $x=\psi(\imath)$, 即所谓第二类换元法.

## 二、第二类换元法

上面介绍的第一类换元法是通过变量代换 $u=\varphi(x)$, 将积分 $\int f[\varphi(x)]$ $\varphi^{\prime}(x) \mathrm{d} x$ 化为积分 $\int f(u) \mathrm{d} u$.

下面将介绍的第二类换元法是: 适当地选择变量代换 $x=\psi(t)$, 将积分 $\int f(x) \mathrm{d} x$ 化为积分 $\int f[\psi(t)] \psi^{\prime}(t) \mathrm{d} t$. 这是另一种形式的变量代换, 换元公式 可表达为

$$
\int f(x) \mathrm{d} x=\int f[\psi(t)] \psi^{\prime}(t) \mathrm{d} t .
$$

这公式的成立是需要一定条件的. 首先, 等式右边的不定积分要存在, 即 $f[\psi(t)] \psi^{\prime}(t)$ 有原函数; 其次, $\int f[\psi(t)] \psi^{\prime}(t) \mathrm{d} t$ 求出后必须用 $x=\psi(t)$ 的反 函数 $\iota=\psi^{-1}(x)$ 代回去, 为了保证这反函数存在而且是可导的, 我们假定直接 函数 $x=\psi(t)$ 在 $t$ 的某一个区间 (这区间和所考虑的 $x$ 的积分区间相对应)上 是单调的、可导的,并且 $\psi^{\prime}(t) \neq 0$.

归纳上述,我们给出下面的定理. 定理 2 设 $x=\psi(t)$ 是单调的、可导的函数, 并且 $\psi^{\prime}(t) \neq 0$. 又设 $f[\psi(t)]$ $\psi^{\prime}(t)$ 具有原函数, 则有换元公式

$$
\int f(x) \mathrm{d} x=\left[\int f[\psi(t)] \psi^{\prime}(t) \mathrm{d} t\right]_{t=\psi^{-1}(x)},
$$

其中 $\psi^{-1}(x)$ 是 $x=\psi(t)$ 的反函数.

证 设 $f[\psi(t)] \psi^{\prime}(t)$ 的原函数为 $\Phi(t)$, 记 $\Phi\left[\psi^{-1}(x)\right]=F(x)$, 利用复 合函数及反函数的求导法则, 得到

$$
\begin{aligned}
F^{\prime}(x) & =\frac{\mathrm{d} \Phi}{\mathrm{d} t} \cdot \frac{\mathrm{d} t}{\mathrm{~d} x}=f[\psi(t)] \psi^{\prime}(t) \cdot \frac{1}{\psi^{\prime}(t)} \\
& =f[\psi(t)]=f(x),
\end{aligned}
$$

即 $F(x)$ 是 $f(x)$ 的原函数. 所以有

$$
\begin{aligned}
\int f(x) \mathrm{d} x & =F(x)+C=\Phi\left[\psi^{-1}(x)\right]+C \\
& =\left[\int f[\psi(t)] \psi^{\prime}(t) \mathrm{d} t\right]_{t=\psi^{-1}(x)}
\end{aligned}
$$

这就证明了公式(2).

下面举例说明换元公式(2)的应用.

例 21 求 $\int \sqrt{a^{2}-x^{2}} \mathrm{~d} x \quad(a>0)$.

解 求这个积分的困难在于有根式 $\sqrt{a^{2}-x^{2}}$, 但我们可以利用三角公式

$$
\sin ^{2} t+\cos ^{2} t=1
$$

来化去根式.

设 $x=a \sin t,-\frac{\pi}{2}<t<\frac{\pi}{2}$, 那么 $\sqrt{a^{2}-x^{2}}=\sqrt{a^{2}-a^{2} \sin ^{2} t}=a \cos t, \mathrm{~d} x=$ $a \cos t \mathrm{~d} t$, 于是根式化成了三角式,所求积分化为

利用例 14 的结果得

$$
\int \sqrt{a^{2}-x^{2}} \mathrm{~d} x=\int a \cos t \cdot a \cos t \mathrm{~d} t=a^{2} \int \cos ^{2} t \mathrm{~d} t .
$$

$$
\begin{aligned}
\int \sqrt{a^{2}-x^{2}} \mathrm{~d} x & =a^{2}\left(\frac{t}{2}+\frac{\sin 2 t}{4}\right)+C \\
& =\frac{a^{2}}{2} t+\frac{a^{2}}{2} \sin t \cos t+C .
\end{aligned}
$$

由于 $x=a \sin t,-\frac{\pi}{2}<t<\frac{\pi}{2}$, 所以

$$
\begin{gathered}
t=\arcsin \frac{x}{a}, \\
\cos t=\sqrt{1-\sin ^{2} t}=\sqrt{1-\left(\frac{x}{a}\right)^{2}}=\frac{\sqrt{a^{2}-x^{2}}}{a},
\end{gathered}
$$

于是所求积分为

$$
\int \sqrt{a^{2}-x^{2}} \mathrm{~d} x=\frac{a^{2}}{2} \arcsin \frac{x}{a}+\frac{1}{2} x \sqrt{a^{2}-x^{2}}+C .
$$

例 22 求 $\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}} \quad(a>0)$.

解 和上例类似, 可以利用三角公式

$$
1+\tan ^{2} t=\sec ^{2} t
$$

来化去根式.

设 $x=a \tan t\left(-\frac{\pi}{2}<t<\frac{\pi}{2}\right)$, 那么

$$
\sqrt{x^{2}+a^{2}}=\sqrt{a^{2}+a^{2} \tan ^{2} t}=a \sqrt{1+\tan ^{2} t}=a \sec t, \mathrm{~d} x=a \sec ^{2} t \mathrm{~d} t,
$$

于是

$$
\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}}=\int \frac{a \sec ^{2} t}{a \sec t} \mathrm{~d} t=\int \sec t \mathrm{~d} t .
$$

利用例 19 的结果得

$$
\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}}=\ln |\sec t+\tan t|+C .
$$

为了要把 $\sec t$ 及 $\tan t$ 换成 $x$ 的函数, 可以根据 $\tan t=\frac{x}{a}$ 作辅助三角形 (图 4-3), 便有

$$
\sec t=\frac{\sqrt{x^{2}+a^{2}}}{a},
$$

且 $\sec t+\tan t>0$, 因此,

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}} & =\ln \left(\frac{x}{a}+\frac{\sqrt{x^{2}+a^{2}}}{a}\right)+C \\
& =\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C_{1},
\end{aligned}
$$

其中 $C_{1}=C-\ln a$.

例 23 求 $\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}}(a>0)$.

解 和以上两例类似, 可以利用公式

$$
\sec ^{2} t-1=\tan ^{2} t
$$

来化去根式. 注意到被积函数的定义域是 $x>a$ 和 $x<-a$ 两个区间, 我们在两 个区间内分别求不定积分.

当 $x>a$ 时, 设 $x=a \sec t\left(0<t<\frac{\pi}{2}\right)$, 那么

$$
\begin{gathered}
\sqrt{x^{2}-a^{2}}=\sqrt{a^{2} \sec ^{2} t-a^{2}}=a \sqrt{\sec ^{2} t-1}=a \tan t, \\
\mathrm{~d} x=a \sec t \tan t \mathrm{~d} t,
\end{gathered}
$$

于是

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}} & =\int \frac{a \sec t \tan t}{a \tan t} \mathrm{~d} t=\int \sec t \mathrm{~d} t \\
& =\ln (\sec t+\tan t)+C .
\end{aligned}
$$

为了把 $\sec t$ 及 $\tan t$ 换成 $x$ 的函数，我们根据 $\sec t=\frac{x}{a}$ 作辅助三角形(图 $4-4)$, 得到

$$
\tan t=\frac{\sqrt{x^{2}-a^{2}}}{a}
$$

因此

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}} & =\ln \left(\frac{x}{a}+\frac{\sqrt{x^{2}-a^{2}}}{a}\right)+C \\
& =\ln \left(x+\sqrt{x^{2}-a^{2}}\right)+C_{1},
\end{aligned}
$$

其中 $C_{1}=C-\ln a$.

当 $x<-a$ 时, 令 $x=-u$, 那么 $u>a$. 由上段 结果, 有

国 4-4

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}} & =-\int \frac{\mathrm{d} u}{\sqrt{u^{2}-a^{2}}}=-\ln \left(u+\sqrt{u^{2}-a^{2}}\right)+C \\
& =-\ln \left(-x+\sqrt{x^{2}-a^{2}}\right)+C \\
& =\ln \frac{-x-\sqrt{x^{2}-a^{2}}}{a^{2}}+C \\
& =\ln \left(-x-\sqrt{x^{2}-a^{2}}\right)+C_{1},
\end{aligned}
$$

其中 $C_{1}=C-2 \ln a$.

把在 $x>a$ 及 $x<-a$ 内的结果合起来, 可写作

$$
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}}=\ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C .
$$

从上面的三个例子可以看出: 如果被积函数含有 $\sqrt{a^{2}-x^{2}}$, 可以作代换 $x=a \sin t$ 化去根式; 如果被积函数含有 $\sqrt{x^{2}+a^{2}}$, 可以作代换 $x=a \tan t$ 化去 根式; 如果被积函数含有 $\sqrt{x^{2}-a^{2}}$, 可以作代换 $x= \pm a \sec t$ 化去根式. 但具体 解题时要分析被积函数的具体情况,选取尽可能简捷的代换,不要拘泥于上述的 变量代换 (如例 5、例 7). 当被积函数含有 $\sqrt{x^{2} \pm a^{2}}$ 时, 为了化去根式, 除采用三角代换 $x=a \tan t$ 或 $x= \pm a \sec t$ 外, 还可利用公式

$$
\operatorname{ch}^{2} t-\operatorname{sh}^{2} t=1 \text {, }
$$

采用双曲代换 $x=a \operatorname{sh~} t 、 x= \pm a$ ch $t$ 来化去根式.

例如,在例 22 中, 可设 $x=a \operatorname{sh} t$, 那么

$$
\sqrt{x^{2}+a^{2}}=\sqrt{a^{2} \operatorname{sh}^{2} t+a^{2}}=a \operatorname{ch} t, \mathrm{~d} x=a \operatorname{ch} t \mathrm{~d} t,
$$

于是

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}} & =\int \frac{a \operatorname{ch} t}{a \operatorname{ch} t} \mathrm{~d} t=\int \mathrm{d} t=t+C \\
& =\operatorname{arsh} \frac{x}{a}+C \\
& =\ln \left[\frac{x}{a}+\sqrt{\left(\frac{x}{a}\right)^{2}+1}\right]+C \\
& =\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C_{1},
\end{aligned}
$$

其中 $C_{1}=C-\ln a$.

在例 23 中, 当 $x>a$ 时, 可设 $x=a \operatorname{ch~} t(t>0)$, 那么

$$
\begin{gathered}
\sqrt{x^{2}-a^{2}}=\sqrt{a^{2} \operatorname{ch}^{2} t-a^{2}}=a \operatorname{sh} t, \\
\mathrm{~d} x=a \operatorname{sh} t \mathrm{~d} t,
\end{gathered}
$$

于是当 $x>a$ 时,

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}} & =\int \frac{a \operatorname{sh} t}{a \operatorname{sh} t} \mathrm{~d} t=\int \mathrm{d} t=t+C \\
& =\operatorname{arch} \frac{x}{a}+C \\
& =\ln \left[\frac{x}{a}+\sqrt{\left(\frac{x}{a}\right)^{2}-1}\right]+C \\
& =\ln \left(x+\sqrt{x^{2}-a^{2}}\right)+C_{1},
\end{aligned}
$$

其中 $C_{1}=C-\ln a$.

当 $x<-a$ 时, 令 $x=-a \operatorname{ch} t(t>0)$, 类似可得

$$
\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}}=\ln \left(-x-\sqrt{x^{2}-a^{2}}\right)+C_{1} .
$$

上节所列基本积分表中没有双曲函数的积分公式,现添加两个常用的双曲 函数积分公式:

$$
\text { (14) } \int \operatorname{sh} x \mathrm{~d} x=\operatorname{ch} x+C \text {, }
$$

(15) $\int \operatorname{ch} x \mathrm{~d} x=\operatorname{sh} x+C$.

下面我们通过例子来介绍一种也很有用的代换一一倒代换,利用它常可消 去被积函数的分母中的变量因子 $x$.

例 24 求 $\int \frac{\sqrt{a^{2}-x^{2}}}{x^{4}} \mathrm{~d} x$.

解 设 $x=\frac{1}{t}$, 那么 $\mathrm{d} x=-\frac{\mathrm{d} t}{t^{2}}$,于是

$$
\begin{aligned}
\int \frac{\sqrt{a^{2}-x^{2}}}{x^{4}} \mathrm{~d} x & =\int \frac{\sqrt{a^{2}-\frac{1}{t^{2}}} \cdot\left(-\frac{\mathrm{d} t}{t^{2}}\right)}{\frac{1}{t^{4}}} \\
& =-\int\left(a^{2} t^{2}-1\right)^{\frac{1}{2}}|t| \mathrm{d} t
\end{aligned}
$$

当 $x>0$ 时,有

$$
\begin{aligned}
\int \frac{\sqrt{a^{2}-x^{2}}}{x^{4}} \mathrm{~d} x & =-\frac{1}{2 a^{2}} \int\left(a^{2} t^{2}-1\right)^{\frac{1}{2}} \mathrm{~d}\left(a^{2} t^{2}-1\right) \\
& =-\frac{\left(a^{2} t^{2}-1\right)^{\frac{3}{2}}}{3 a^{2}}+C \\
& =-\frac{\left(a^{2}-x^{2}\right)^{\frac{3}{2}}}{3 a^{2} x^{3}}+C,
\end{aligned}
$$

当 $x<0$ 时,有相同的结果.

在本节的例题中,有几个积分是以后经常会遇到的.所以它们通常也被当作 公式使用. 这样, 常用的积分公式,除了基本积分表中的几个外,再添加下面几个 (其中常数 $a>0$ )：
(16) $\int \tan x \mathrm{~d} x=-\ln |\cos x|+C$,
(11) $\int \cot x \mathrm{~d} x=\ln |\sin x|+C$,
(18) $\int \sec x \mathrm{~d} x=\ln |\sec x+\tan x|+C$,
(19) $\int \csc x \mathrm{~d} x=\ln |\csc x-\cot x|+C$,
(20) $\int \frac{\mathrm{d} x}{a^{2}+x^{2}}=\frac{1}{a} \arctan \frac{x}{a}+C$,
(21) $\int \frac{\mathrm{d} x}{x^{2}-a^{2}}=\frac{1}{2 a} \ln \left|\frac{x-a}{x+a}\right|+C$, (2) $\int \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}}=\arcsin \frac{x}{a}+C$,

(2) $\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}}=\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$,

(24) $\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}}=\ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$.

例 25 求 $\int \frac{\mathrm{d} x}{\sqrt{4 x^{2}+9}}$.

解 $\int \frac{\mathrm{d} x}{\sqrt{4 x^{2}+9}}=\int \frac{\mathrm{d} x}{\sqrt{(2 x)^{2}+3^{2}}}=\frac{1}{2} \int \frac{\mathrm{d}(2 x)}{\sqrt{(2 x)^{2}+3^{2}}}$,

利用公式(3)，便得

$$
\int \frac{\mathrm{d} x}{\sqrt{4 x^{2}+9}}=\frac{1}{2} \ln \left(2 x+\sqrt{4 x^{2}+9}\right)+C .
$$

例 26 求 $\int \frac{\mathrm{d} x}{\sqrt{1+x-x^{2}}}$.

解 $\int \frac{\mathrm{d} x}{\sqrt{1+x-x^{2}}}=\int \frac{\mathrm{d}\left(x-\frac{1}{2}\right)}{\sqrt{\left(\frac{\sqrt{5}}{2}\right)^{2}-\left(x-\frac{1}{2}\right)^{2}}}$,

利用公式，便得

$$
\int \frac{\mathrm{d} x}{\sqrt{1+x-x^{2}}}=\arcsin \frac{2 x-1}{\sqrt{5}}+C .
$$

在例 22 中,我们用变换 $x=a \tan t$ 消去被积函数中的根式 $\sqrt{x^{2}+a^{2}}$, 这个 变换还能消去被积函数分母中的 $\left(x^{2}+a^{2}\right)$ 的高次幂. 请看下例.

例 27 求 $\int \frac{x^{3}}{\left(x^{2}-2 x+2\right)^{2}} \mathrm{~d} x$.

解 分母是二次质因式的平方, 把二次质因式配方成 $(x-1)^{2}+1$, 令 $x-1$ $=\tan t\left(-\frac{\pi}{2}<t<\frac{\pi}{2}\right)$, 则

$$
x^{2}-2 x+2=\sec ^{2} t, \mathrm{~d} x=\sec ^{2} t \mathrm{~d} t .
$$

于是 $\quad \int \frac{x^{3}}{\left(x^{2}-2 x+2\right)^{2}} \mathrm{~d} x$

$$
\begin{aligned}
& =\int \frac{(\tan t+1)^{3}}{\sec ^{4} t} \cdot \sec ^{2} t \mathrm{~d} t \\
& =\int\left(\sin ^{3} t \cos ^{-1} t+3 \sin ^{2} t+3 \sin t \cos t+\cos ^{2} t\right) \mathrm{d} t
\end{aligned}
$$

$$
\begin{aligned}
& =\int\left(\sin ^{2} t \cos ^{-1} t+3 \cos t\right) \sin t \mathrm{~d} t+\int\left(3 \sin ^{2} t+\cos ^{2} t\right) \mathrm{d} t \\
& =\int\left[\left(1-\cos ^{2} t\right) \cos ^{-1} t+3 \cos t\right][-\mathrm{d}(\cos t)]+\int(2-\cos 2 t) \mathrm{d} t \\
& =-\int\left(\cos ^{-1} t+2 \cos t\right) \mathrm{d}(\cos t)+2 t-\frac{1}{2} \sin 2 t \\
& =-\ln \cos t-\cos ^{2} t+2 t-\sin t \cos t+C,
\end{aligned}
$$

按 $\tan t=x-1$ 作辅助三角形(图 4-5), 便有 $\cos t=\frac{1}{\sqrt{x^{2}-2 x+2}}, \sin t=\frac{x-1}{\sqrt{x^{2}-2 x+2}}$,

于是

$$
\begin{aligned}
& \int \frac{x^{3}}{\left(x^{2}-2 x+2\right)^{2}} \mathrm{~d} x \\
= & \frac{1}{2} \ln \left(x^{2}-2 x+2\right)+2 \arctan (x-1)-\frac{x}{x^{2}-2 x+2}+C .
\end{aligned}
$$

## 习 题 4-2

1. 在下列各式等号右端的空白处填入适当的系数，健等式成立(例如: $\mathrm{d} x=\frac{1}{4} \mathrm{~d}(4 x+7)$ ):
(1) $\mathrm{d} x=\mathrm{d}(a x)$;
(2) $\mathrm{d} x=\mathrm{d}(7 x-3)$;
(3) $x \mathrm{~d} x=\mathrm{d}\left(x^{2}\right)$;
(4) $x \mathrm{~d} x=\mathrm{d}\left(5 x^{2}\right)$;
(5) $x \mathrm{~d} x=\mathrm{d}\left(1-x^{2}\right)$;
(6) $x^{3} \mathrm{~d} x=\mathrm{d}\left(3 x^{4}-2\right)$;
(7) $\mathrm{e}^{2 x} \mathrm{~d} x=\mathrm{d}\left(\mathrm{e}^{2 . x}\right)$;
(8) $\mathrm{e}^{-\frac{x}{2}} \mathrm{~d} x=\mathrm{d}\left(1+\mathrm{e}^{-\frac{x}{2}}\right)$;
(9) $\sin \frac{3}{2} x \mathrm{~d} x=\mathrm{d}\left(\cos \frac{3}{2} x\right)$;
(10) $\frac{d x}{x}=d(5 \ln |x|)$;
(11) $\frac{\mathrm{d} x}{x}=\mathrm{d}(3-5 \ln |x|)$;
(12) $\frac{\mathrm{d} x}{1+9 x^{2}}=\mathrm{d}(\arctan 3 x)$;
(13) $\frac{\mathrm{d} x}{\sqrt{1-x^{2}}}=\mathrm{d}(1-\arcsin x)$;
(14) $\frac{x \mathrm{~d} x}{\sqrt{1-x^{2}}}=\mathrm{d}\left(\sqrt{1-x^{2}}\right)$.
2. 求下列不定积分 (其非 $a, b 、 \omega 、 \varphi$ 均为学数):
(1) $\int \mathrm{e}^{3 t} \mathrm{~d} t$;
(2) $\int(3-2 . x)^{3} \mathrm{~d} x$;
(3) $\int \frac{\mathrm{d} x}{1-2 . x}$;
(4) $\int \frac{\mathrm{d} x}{\sqrt[3]{2-3 x}}$;
(5) $\int\left(\sin a x-\mathrm{e}^{\frac{x}{5}}\right) \mathrm{d} x$;
(6) $\int \frac{\sin \sqrt{t}}{\sqrt{t}} \mathrm{~d} t$;
(7) $\int x \mathrm{e}^{-t^{2}} \mathrm{~d} x$;
(8) $\int x \cos \left(x^{2}\right) \mathrm{d} x$; (9) $\int \frac{x}{\sqrt{2-3 x^{2}}} \mathrm{~d} x$;

(11) $\int \frac{x+1}{x^{2}+2 x+5} \mathrm{~d} x$;

(13) $\int \frac{\sin x}{\cos ^{3} x} \mathrm{~d} x$;

(15) $\int \tan ^{10} x \cdot \sec ^{2} x \mathrm{~d} x$;

(17) $\int \frac{\mathrm{d} x}{(\arcsin x)^{2} \sqrt{1-x^{2}}}$;

(19) $\int \tan \sqrt{1+x^{2}} \cdot \frac{x \mathrm{~d} x}{\sqrt{1+x^{2}}}$;

(21) $\int \frac{1+\ln x}{(x \ln x)^{2}} \mathrm{~d} x$;

(23) $\int \frac{\ln \tan x}{\cos x \sin x} \mathrm{~d} x$;

(25) $\int \cos ^{2}(\omega t+\varphi) \mathrm{d} t$;

(27) $\int \cos x \cos \frac{x}{2} \mathrm{~d} x$;

(29) $\int \tan ^{3} x \sec x \mathrm{~d} x$;

(31) $\int \frac{1-x}{\sqrt{9-4 x^{2}}} \mathrm{~d} x$;

(33) $\int \frac{\mathrm{d} x}{2 x^{2}-1}$;

(35) $\int \frac{x}{x^{2}-x-2} \mathrm{~d} x$;

(37) $\int \frac{\mathrm{d} x}{x \sqrt{x^{2}-1}}$;

(39) $\int \frac{\sqrt{x^{2}-9}}{x} \mathrm{~d} x$;

(41) $\int \frac{\mathrm{d} x}{1+\sqrt{1-x^{2}}}$;

(43) $\int \frac{x-1}{x^{2}+2 x+3} \mathrm{~d} x$;
(10) $\int \frac{3 x^{3}}{1-x^{4}} \mathrm{~d} x$;

(12) $\int \cos ^{2}(\omega t+\varphi) \sin (\omega t+\varphi) \mathrm{d} t$;

(14) $\int \frac{\sin x+\cos x}{\sqrt[3]{\sin x-\cos x}} \mathrm{~d} x$;

(16) $\int \frac{\mathrm{d} x}{x \ln x \ln \ln x}$;

(18) $\int \frac{10^{2 \text { nnam } . x}}{\sqrt{1-x^{2}}} \mathrm{~d} x$;

(20) $\int \frac{\arctan \sqrt{x}}{\sqrt{x}(1+x)} \mathrm{d} x$;

(22) $\int \frac{\mathrm{d} x}{\sin x \cos x}$;

(24) $\int \cos ^{3} x \mathrm{~d} x$

(26) $\int \sin 2 x \cos 3 x \mathrm{~d} x$;

(28) $\int \sin 5 x \sin 7 x \mathrm{~d} x$;

(30) $\int \frac{\mathrm{d} x}{\mathrm{e}^{x}+\mathrm{e}^{-x}}$;

(32) $\int \frac{x^{3}}{9+x^{2}} \mathrm{~d} x$;

(34) $\int \frac{\mathrm{d} x}{(x+1)(x-2)}$;

(36) $\int \frac{x^{2} \mathrm{~d} x}{\sqrt{a^{2}-x^{2}}} \quad(a>0)$;

(38) $\int \frac{\mathrm{d} x}{\sqrt{\left(x^{2}+1\right)^{3}}}$;

(40) $\int \frac{\mathrm{d} x}{1+\sqrt{2 x}}$;

(42) $\int \frac{\mathrm{d} x}{x+\sqrt{1-x^{2}}}$;

(44) $\int \frac{x^{3}+1}{\left(x^{2}+1\right)^{2}} \mathrm{~d} x$.

## 第三节 分部积分法

前面我们在复合函数求导法则的基础上,得到了换元积分法. 现在我们利用 两个函数乘积的求导法则, 来推得另一个求积分的基本方法一一分部积分法. 设函数 $u=u(x)$ 及 $v=v(x)$ 具有连续导数. 那么, 两个函数乘积的导数公 式为
移项，得

$$
\begin{aligned}
& (u v)^{\prime}=u^{\prime} v+u v^{\prime}, \\
& u v^{\prime}=(u v)^{\prime}-u^{\prime} v .
\end{aligned}
$$

对这个等式两边求不定积分, 得

$$
\int u v^{\prime} \mathrm{d} x=u v-\int u^{\prime} v \mathrm{~d} x .
$$

公式 (1) 称为分部积分公式. 如果求 $\int u v^{\prime} \mathrm{d} x$ 有困难, 而求 $\int u^{\prime} v \mathrm{~d} x$ 比较容易时, 分部积分公式就可以发挥作用了.

为简便起见,也可把公式(1)写成下面的形式:

$$
\int u \mathrm{~d} v=u v-\int v \mathrm{~d} u .
$$

现在通过例子说明如何运用这个重要公式.

例 1 求 $\int x \cos x \mathrm{~d} x$.

解 这个积分用换元积分法不易求得结果. 现在试用分部积分法来求它. 但 是怎样选取 $u$ 和 $\mathrm{d} v$ 呢? 如果设 $u=x, \mathrm{~d} v=\cos x \mathrm{~d} x$, 那么 $\mathrm{d} u=\mathrm{d} x, v=\sin x$ ， 代入分部积分公式(2), 得

$$
\int x \cos x \mathrm{~d} x=x \sin x-\int \sin x \mathrm{~d} x,
$$

而 $\int v \mathrm{~d} u=\int \sin x \mathrm{~d} x$ 容易积出, 所以

$$
\int x \cos x \mathrm{~d} x=x \sin x+\cos x+C .
$$

求这个积分时, 如果设 $u=\cos x, \mathrm{~d} v=x \mathrm{~d} x$, 那么

$$
\mathrm{d} u=-\sin x \mathrm{~d} x, v=\frac{x^{2}}{2} \text {. }
$$

于是 $\quad \int x \cos x \mathrm{~d} x=\frac{x^{2}}{2} \cos x+\int \frac{x^{2}}{2} \sin x \mathrm{~d} x$.

上式右端的积分比原积分更不容易求出.

由此可见,如果 $u$ 和 $\mathrm{d} v$ 选取不当, 就求不出结果, 所以应用分部积分法时, 恰当选取 $u$ 和 $\mathrm{d} v$ 是一个关键. 选取 $u$ 和 $\mathrm{d} v$ 一般要考虑下面两点:
（1） $v$ 要容易求得;
(2) $\int v \mathrm{~d} u$ 要比 $\int u \mathrm{~d} v$ 容易积出.

例 2 求 $\int x \mathrm{e}^{\mathrm{x}} \mathrm{d} x$. 解 设 $u=x, \mathrm{~d} v=\mathrm{e}^{x} \mathrm{~d} x$, 那么 $\mathrm{d} u=\mathrm{d} x, v=\mathrm{e}^{x}$. 于是

$$
\int x \mathrm{e}^{x} \mathrm{~d} x=x \mathrm{e}^{x}-\int \mathrm{e}^{x} \mathrm{~d} x=x \mathrm{e}^{x}-\mathrm{e}^{x}+C=\mathrm{e}^{\mathrm{r}}(x-1)+C .
$$

运用分部积分公式 (2)的形式, 如上列例 1 、例 2 的求解过程也可表述为

$$
\begin{aligned}
\int x \cos x \mathrm{~d} x & =\int x \mathrm{~d}(\sin x)=x \sin x-\int \sin x \mathrm{~d} x \\
& =x \sin x+\cos x+C . \\
\int x \mathrm{e}^{x} \mathrm{~d} x & =\int x \mathrm{~d}\left(\mathrm{e}^{x}\right)=x \mathrm{e}^{x}-\int \mathrm{e}^{x} \mathrm{~d} x \\
& =x \mathrm{e}^{x}-\mathrm{e}^{x}+C=(x-1) \mathrm{e}^{x}+C .
\end{aligned}
$$

例 3 求 $\int x^{2} \mathrm{e}^{x} \mathrm{~d} x$.

解 设 $u=x^{2}, \mathrm{~d} v=\mathrm{e}^{\mathrm{x}} \mathrm{d} x=\mathrm{d}\left(\mathrm{e}^{x}\right)$, 那么

$$
\int x^{2} \mathrm{e}^{\mathrm{r}} \mathrm{d} x=\int x^{2} \mathrm{~d}\left(\mathrm{e}^{-x}\right)=x^{2} \mathrm{e}^{\mathrm{r}}-\int \mathrm{e}^{x} \mathrm{~d}\left(x^{2}\right)=x^{2} \mathrm{e}^{x}-2 \int x \mathrm{e}^{\mathrm{x}} \mathrm{d} x .
$$

这里 $\int x \mathrm{e}^{x} \mathrm{~d} x$ 比 $\int x^{2} \mathrm{e}^{x} \mathrm{~d} x$ 容易积出, 因为被积函数中 $x$ 的军次前者比后 者降低了一次. 由例 2 可知, 对 $\int x \mathrm{e}^{\mathrm{r}} \mathrm{d} x$ 再使用一次分部积分法就可以了.于是

$$
\begin{aligned}
\int x^{2} \mathrm{e}^{x} \mathrm{~d} x & =x^{2} \mathrm{e}^{x}-2 \int x \mathrm{e}^{x} \mathrm{~d} x=x^{2} \mathrm{e}^{x}-2 \int x \mathrm{~d}\left(\mathrm{e}^{x}\right) \\
& =x^{2} \mathrm{e}^{x}-2\left(x \mathrm{e}^{x}-\mathrm{e}^{x}\right)+C \\
& =\mathrm{e}^{x}\left(x^{2}-2 x+2\right)+C .
\end{aligned}
$$

总结上面三个例子可以知道, 如果被积函数是幂函数和正 (余) 弦函数或幂函数和指数函数的乘积, 就可以考虑用分部积分法, 并设幂函数为 $u$. 这样用一 次分部积分法就可以使幂函数的幂次降低一次. 这里假定幂指数是正整数.

例 4 求 $\int x \ln x \mathrm{~d} x$.

解 设 $u=\ln x, \mathrm{~d} v=x \mathrm{~d} x$, 那么

$$
\begin{aligned}
\int x \ln x \mathrm{~d} x & =\int \ln x \mathrm{~d} \frac{x^{2}}{2} \\
& =\frac{x^{2}}{2} \ln x-\int \frac{x^{2}}{2} \mathrm{~d}(\ln x) \\
& =\frac{x^{2}}{2} \ln x-\frac{1}{2} \int x \mathrm{~d} x \\
& =\frac{x^{2}}{2} \ln x-\frac{x^{2}}{4}+C .
\end{aligned}
$$

例 5 求 $\int \arccos x \mathrm{~d} x$. 解 设 $u=\arccos x, \mathrm{~d} v=\mathrm{d} x$, 那么

$$
\begin{aligned}
\int \arccos x \mathrm{~d} x & =x \arccos x-\int x \mathrm{~d}(\arccos x) \\
& =x \arccos x+\int \frac{x}{\sqrt{1-x^{2}}} \mathrm{~d} x \\
& =x \arccos x-\frac{1}{2} \int \frac{1}{\left(1-x^{2}\right)^{\frac{1}{2}}} \mathrm{~d}\left(1-x^{2}\right) \\
& =x \arccos x-\sqrt{1-x^{2}}+C .
\end{aligned}
$$

在分部积分法运用比较熟练以后, 就不必再写出哪一部分选作 $u$, 哪一部分 选作 $\mathrm{d} v$. 只要把被积表达式凑成 $\varphi(x) \mathrm{d} \psi(x)$ 的形式,便可使用分部积分公式.

例 6 求 $\int x \arctan x \mathrm{~d} x$.

$$
\text { 解 } \begin{aligned}
\int x \arctan x \mathrm{~d} x & =\frac{1}{2} \int \arctan x \mathrm{~d}\left(x^{2}\right) \\
& =\frac{x^{2}}{2} \arctan x-\frac{1}{2} \int \frac{x^{2}}{1+x^{2}} \mathrm{~d} x \\
& =\frac{x^{2}}{2} \arctan x-\frac{1}{2} \int \frac{1+x^{2}-1}{1+x^{2}} \mathrm{~d} x \\
& =\frac{x^{2}}{2} \arctan x-\frac{1}{2} \int\left(1-\frac{1}{1+x^{2}}\right) \mathrm{d} x \\
& =\frac{x^{2}}{2} \arctan x-\frac{1}{2}(x-\arctan x)+C \\
& =\frac{1}{2}\left(x^{2}+1\right) \arctan x-\frac{1}{2} x+C .
\end{aligned}
$$

总结上面三个例子可以知道,如果被积函数是幂函数和对数函数或算函数 和反三角函数的乘积, 就可以考虑用分部积分法, 并设对数函数或反三角函数为 $u$.

下面几个例子中所用的方法也是比较典型的．

例 7 求 $\int \mathrm{e}^{x} \sin x \mathrm{~d} x$.

解 $\int \mathrm{e}^{x} \sin x \mathrm{~d} x=\int \sin x \mathrm{~d}\left(\mathrm{e}^{x}\right)=\mathrm{e}^{x} \sin x-\int \mathrm{e}^{\mathrm{x}} \cos x \mathrm{~d} x$,

等式右端的积分与等式左端的积分是同一类型的.对右端的积分再用一次分部 积分法,得

$$
\begin{aligned}
& \int \mathrm{e}^{x} \sin x \mathrm{~d} x=\mathrm{e}^{x} \sin x-\int \cos x \mathrm{~d}\left(\mathrm{e}^{x}\right) . \\
& =\mathrm{e}^{\mathrm{x}} \sin x-\mathrm{e}^{x} \cos x-\int \mathrm{e}^{x} \sin x \mathrm{~d} x,
\end{aligned}
$$

由于上式右端的第三项就是所求的积分 $\int \mathrm{e}^{x} \sin x \mathrm{~d} x$, 把它移到等号左端去, 再 两端同除以 2 , 便得

$$
\int \mathrm{e}^{x} \sin x \mathrm{~d} x=\frac{1}{2} \mathrm{e}^{x}(\sin x-\cos x)+C .
$$

因上式右端已不包含积分项,所以必须加上任意常数 $C$.

例 8 求 $\int \sec ^{3} x \mathrm{~d} x$.

$$
\text { 解 } \begin{aligned}
\int \sec ^{3} x \mathrm{~d} x & =\int \sec x \mathrm{~d}(\tan x) \\
& =\sec x \tan x-\int \sec x \tan ^{2} x \mathrm{~d} x \\
& =\sec x \tan x-\int \sec x\left(\sec ^{2} x-1\right) \mathrm{d} x \\
& =\sec x \tan x-\int \sec ^{3} x \mathrm{~d} x+\int \sec x \mathrm{~d} x \\
& =\sec x \tan x+\ln |\sec x+\tan x|-\int \sec ^{3} x \mathrm{~d} x .
\end{aligned}
$$

由于上式右端的第三项就是所求的积分 $\int \sec ^{3} x \mathrm{~d} x$, 把它移到等号左端去, 再两 端各除以 2 , 便得

$$
\int \sec ^{3} x \mathrm{~d} x=\frac{1}{2}(\sec x \tan x+\ln |\sec x+\tan x|)+C .
$$

在积分的过程中往往要菲用换元法与分部积分法, 如例 5 ,下面再来举一个 例子.

例 9 求 $\int \mathrm{e}^{\sqrt{\cdot}} \mathrm{d} x$.

解 令 $\sqrt{x}=t$, 则 $x=t^{2}, \mathrm{~d} x=2 t \mathrm{~d} t$. 于是

$$
\int \mathrm{e}^{\sqrt{x}} \mathrm{~d} x=2 \int t \mathrm{e}^{t} \mathrm{~d} t
$$

利用例 2 的结果, 并用 $t=\sqrt{x}$ 代回, 便得所求积分:

$$
\begin{aligned}
\int \mathrm{e}^{\sqrt{x}} \mathrm{~d} x & =2 \int t \mathrm{e}^{t} \mathrm{~d} t=2 \mathrm{e}^{t}(t-1)+C \\
& =2 \mathrm{e}^{\sqrt{x}}(\sqrt{x}-1)+C .
\end{aligned}
$$

## 习 题 4-3

求下列不定积分:

1. $\int x \sin x \mathrm{~d} x$.
2. $\int \ln x \mathrm{~d} x$.
3. $\int \arcsin x \mathrm{~d} x$.
4. $\int x \mathrm{e}^{-x} \mathrm{~d} x$.
5. $\int x^{2} \ln x \mathrm{~d} x$.
6. $\int \mathrm{e}^{-x} \cos x \mathrm{~d} x$.
7. $\int \mathrm{e}^{-2 x} \sin \frac{x}{2} \mathrm{~d} x$.
8. $\int x \cos \frac{x}{2} \mathrm{~d} x$.
9. $\int x^{2} \arctan x \mathrm{~d} x$.
10. $\int x \tan ^{2} x \mathrm{~d} x$.
11. $\int x^{2} \cos x \mathrm{~d} x$.
12. $\int t \mathrm{e}^{-2 t} \mathrm{~d} t$.
13. $\int \ln ^{2} x \mathrm{~d} x$.
14. $\int x \sin x \cos x \mathrm{~d} x$.
15. $\int x^{2} \cos ^{2} \frac{x}{2} \mathrm{~d} x$.
16. $\int x \ln (x-1) \mathrm{d} x$.
17. $\int\left(x^{2}-1\right) \sin 2 x \mathrm{~d} x$.
18. $\int \frac{\ln ^{3} x}{x^{2}} \mathrm{~d} x$.
19. $\int \mathrm{e}^{\sqrt[3]{x}} \mathrm{~d} x$.
20. $\int \cos \ln x \mathrm{~d} x$.
21. $\int(\arcsin x)^{2} \mathrm{~d} x$.
22. $\int \mathrm{e}^{x} \sin ^{2} x \mathrm{~d} x$.
23. $\int x \ln ^{2} x \mathrm{~d} x$.
24. $\int \mathrm{e}^{\sqrt{3 x+9}} \mathrm{~d} x$.

## 第四节 有理函数的积分

前面已经介绍了求不定积分的两个基本方法一一换元积分法和分部积分 法.下面简要地介绍有理函数的积分及可化为有理函数的积分.

## 一、有理函数的积分

两个多项式的商 $\frac{P(x)}{Q(x)}$ 称为有理函数, 又称有理分式. 我们总假定分子多项 式 $P(x)$ 与分母多项式 $Q(x)$ 之间是没有公因式的. 当分子多项式 $P(x)$ 的次数 小于分母多项式 $Q(x)$ 的次数时,称这有理函数为真分式,否则称为假分式.

利用多项式的除法, 总可以将一个假分式化成一个多项式与一个真分式之 和的形式,例如第一节例 15 中的被积函数

$$
\frac{2 x^{4}+x^{2}+3}{x^{2}+1}=2 x^{2}-1+\frac{4}{x^{2}+1} \text {. }
$$

对于真分式 $\frac{P(x)}{Q(x)}$, 如果分母可分解为两个多项式的乘积

$$
Q(x)=Q_{1}(x) Q_{2}(x),
$$

且 $Q_{1}(x)$ 与 $Q_{2}(x)$ 没有公因式, 那么它可分拆成两个真分式之和

$$
\frac{P(x)}{Q(x)}=\frac{P_{1}(x)}{Q_{1}(x)}+\frac{P_{2}(x)}{Q_{2}(x)},
$$

上述步骤称为把真分式化成部分分式之和. 如果 $Q_{1}(x)$ 或 $Q_{2}(x)$ 还能再分解成 两个没有公因式的多项式的乘积,那么就可再分拆成更简单的部分分式. 最后, 有理函数的分解式中只出现多项式、 $\frac{P_{1}(x)}{(x-a)^{k}} 、 \frac{P_{2}(x)}{\left(x^{2}+p x+q\right)^{2}}$ 等三类函数 (这里 $p^{2}-4 q<0, P_{1}(x)$ 为小于 $k$ 次的多项式, $P_{2}(x)$ 为小于 $2 l$ 次的多项式). 多项 式的积分容易求得,后两类真分式的积分可参看第二节例 3 和例 27 .

下面举几个真分式的积分的例子.

例 1 求 $\int \frac{x+1}{x^{2}-5 x+6} \mathrm{~d} x$.

解 被积函数的分母分解成 $(x-3)(x-2)$, 故可设

$$
\frac{x+1}{x^{2}-5 x+6}=\frac{A}{x-3}+\frac{B}{x-2},
$$

其中 $A 、 B$ 为待定系数.上式两端去分母后, 得

$$
x+1=A(x-2)+B(x-3) \text {, }
$$

即

$$
x+1=(A+B) x-2 A-3 B \text {. }
$$

比较上式两端同次幂的系数, 即有

从而解得

$$
\left\{\begin{array}{l}
A+B=1, \\
2 A+3 B=-1,
\end{array}\right.
$$

于是

$$
A=4, B=-3 \text {. }
$$

$$
\begin{aligned}
\int \frac{x+1}{x^{2}-5 x+6} \mathrm{~d} x & =\int\left(\frac{4}{x-3}-\frac{3}{x-2}\right) \mathrm{d} x \\
& =4 \ln |x-3|-3 \ln |x-2|+C .
\end{aligned}
$$

例 2 求 $\int \frac{x+2}{(2 x+1)\left(x^{2}+x+1\right)} \mathrm{d} x$.

解 设

$$
\frac{x+2}{(2 x+1)\left(x^{2}+x+1\right)}=\frac{A}{2 x+1}+\frac{B x+C}{x^{2}+x+1},
$$

则

$$
x+2=A\left(x^{2}+x+1\right)+(B x+C)(2 x+1) \text {, }
$$

即

$$
x+2=(A+2 B) x^{2}+(A+B+2 C) x+A+C \text {, }
$$

有

$$
\left\{\begin{array} { l } 
{ A + 2 B = 0 , } \\
{ A + B + 2 C = 1 , } \\
{ A + C = 2 , }
\end{array} \text { 解得 } \left\{\begin{array}{l}
A=2, \\
B=-1, \\
C=0 .
\end{array}\right.\right.
$$

于是 $\cdot \int \frac{x+2}{(2 x+1)\left(x^{2}+x+1\right)} \mathrm{d} x$

$$
\begin{aligned}
& =\int\left(\frac{2}{2 x+1}-\frac{x}{x^{2}+x+1}\right) \mathrm{d} x \\
& =\ln |2 x+1|-\frac{1}{2} \int \frac{(2 x+1)-1}{x^{2}+x+1} \mathrm{~d} x \\
& =\ln |2 x+1|-\frac{1}{2} \int \frac{\mathrm{d}\left(x^{2}+x+1\right)}{x^{2}+x+1}+\frac{1}{2} \int \frac{\mathrm{d} x}{\left(x+\frac{1}{2}\right)^{2}+\frac{3}{4}} \\
& =\ln |2 x+1|-\frac{1}{2} \ln \left(x^{2}+x+1\right)+\frac{1}{\sqrt{3}} \arctan \frac{2 x+1}{\sqrt{3}}+C .
\end{aligned}
$$

例 3 求 $\int \frac{x-3}{(x-1)\left(x^{2}-1\right)} \mathrm{d} x$.

解 被积函数分母的两个因式 $x-1$ 与 $x^{3}-1$ 有公因式,故需再分解成 $(x-1)^{2}(x+1)$. 设

则

即

有

于是

$$
\frac{x-3}{(x-1)^{2}(x+1)}=\frac{A x+B}{(x-1)^{2}}+\frac{C}{x+1},
$$

$$
\begin{gathered}
x-3=(A x+B)(x+1)+C(x-1)^{2}, \\
x-3=(A+C) x^{2}+(A+B-2 C) x+B+C,
\end{gathered}
$$

$$
\left\{\begin{array} { l } 
{ A + C = 0 , } \\
{ A + B - 2 C = 1 , } \\
{ B + C = - 3 , }
\end{array} \text { 解得 } \left\{\begin{array}{l}
A=1, \\
B=-2, \\
C=-1 .
\end{array}\right.\right.
$$

$$
\begin{aligned}
& \int \frac{x-3}{(x-1)\left(x^{2}-1\right)} \mathrm{d} x \\
& =\int \frac{x-3}{(x-1)^{2}(x+1)} \mathrm{d} x \\
& =\int\left[\frac{x-2}{(x-1)^{2}}-\frac{1}{x+1}\right] \mathrm{d} x \\
& =\int \frac{x-1-1}{(x-1)^{2}} \mathrm{~d} x-\ln |x+1| \\
& =\ln |x-1|+\frac{1}{x-1}-\ln |x+1|+C .
\end{aligned}
$$

## 二、可化为有理函数的积分举例

例 4 求 $\int \frac{1+\sin x}{\sin x(1+\cos x)} \mathrm{d} x$. 解 由三角函数知道, $\sin x$ 与 $\cos x$ 都可以用 $\tan \frac{x}{2}$ 的有理式表示, 即

$$
\begin{gathered}
\sin x=2 \sin \frac{x}{2} \cos \frac{x}{2}=\frac{2 \tan \frac{x}{2}}{\sec ^{2} \frac{x}{2}}=\frac{2 \tan \frac{x}{2}}{1+\tan ^{2} \frac{x}{2}}, \\
\cos x=\cos ^{2} \frac{x}{2}-\sin ^{2} \frac{x}{2}=\frac{1-\tan ^{2} \frac{x}{2}}{\sec ^{2} \frac{x}{2}}=\frac{1-\tan ^{2} \frac{x}{2}}{1+\tan ^{2} \frac{x}{2}} .
\end{gathered}
$$

如果作变换 $u=\tan \frac{x}{2}(-\pi<x<\pi)$, 那么

$$
\sin x=\frac{2 u}{1+u^{2}}, \quad \cos x=\frac{1-u^{2}}{1+u^{2}},
$$

而 $x=2 \arctan u$, 从而

$$
\mathrm{d} x=\frac{2}{1+u^{2}} \mathrm{~d} u
$$

于是

$$
\begin{aligned}
& \int \frac{1+\sin x}{\sin x(1+\cos x)} \mathrm{d} x \\
= & \int \frac{\left(1+\frac{2 u}{1+u^{2}}\right) \frac{2 \mathrm{~d} u}{1+u^{2}}}{\frac{2 u}{1+u^{2}}\left(1+\frac{1-u^{2}}{1+u^{2}}\right)} \\
= & \frac{1}{2} \int\left(u+2+\frac{1}{u}\right) \mathrm{d} u \\
= & \frac{1}{2}\left(\frac{u^{2}}{2}+2 u+\ln |u|\right)+C \\
= & \frac{1}{4} \tan ^{2} \frac{x}{2}+\tan \frac{x}{2}+\frac{1}{2} \ln \left|\tan \frac{x}{2}\right|+C .
\end{aligned}
$$

本例所用的变量代换 $u=\tan \frac{x}{2}$ 对三角函数有理式的积分都可以应用.

例 5 求 $\int \frac{\sqrt{x-1}}{x} \mathrm{~d} x$.

解 为了去掉根号, 可以设 $\sqrt{x-1}=u$,于是 $x=u^{2}+1, \mathrm{~d} x=2 u \mathrm{~d} u$, 从而 所求积分为

$$
\begin{aligned}
\int \frac{\sqrt{x-1}}{x} \mathrm{~d} x & =\int \frac{u}{u^{2}+1} \cdot 2 u \mathrm{~d} u=2 \int \frac{u^{2}}{u^{2}+1} \mathrm{~d} u \\
& =2 \int\left(1-\frac{1}{1+u^{2}}\right) \mathrm{d} u=2(u-\arctan u)+C
\end{aligned}
$$

$$
=2(\sqrt{x-1}-\arctan \sqrt{x-1})+C .
$$

例 6 求 $\int \frac{\mathrm{d} x}{1+\sqrt[3]{x+2}}$.

解 为了去掉根号, 可以设 $\sqrt[3]{x+2}=u$.于是 $x=u^{3}-2, \mathrm{~d} x=3 u^{2} \mathrm{~d} u$, 从而 所求积分为

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{1+\sqrt[3]{x+2}} & =\int \frac{3 u^{2}}{1+u} \mathrm{~d} u \\
& =3 \int\left(u-1+\frac{1}{1+u}\right) \mathrm{d} u=3\left(\frac{u^{2}}{2}-u+\ln |1+u|\right)+C \\
& =\frac{3}{2} \sqrt[3]{(x+2)^{2}}-3 \sqrt[3]{x+2}+3 \ln |1+\sqrt[3]{x+2}|+C .
\end{aligned}
$$

例 7 求 $\int \frac{\mathrm{d} x}{(1+\sqrt[3]{x}) \sqrt{x}}$.

解 被积函数中出现了两个根式 $\sqrt{x}$ 及 $\sqrt[3]{x}$. 为了能同时消去这两个根式, 可 令 $x=t^{6}$. 于是 $\mathrm{d} x=6 t^{5} \mathrm{~d} t$, 从而所求积分为

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{(1+\sqrt[3]{x}) \sqrt{x}} & =\int \frac{6 t^{5}}{\left(1+t^{2}\right) t^{3}} \mathrm{~d} t=6 \int \frac{t^{2}}{1+t^{2}} \mathrm{~d} t \\
& =6 \int\left(1-\frac{1}{1+t^{2}}\right) \mathrm{d} t=6(t-\arctan t)+C \\
& =6(\sqrt[6]{x}-\arctan \sqrt[6]{x})+C .
\end{aligned}
$$

例 8 求 $\int \frac{1}{x} \sqrt{\frac{1+x}{x}} \mathrm{~d} x$.

解 为了去掉根号, 可以设 $\sqrt{\frac{1+x}{x}}=t$, 于是 $\frac{1+x}{x}=t^{2}, x=\frac{1}{t^{2}-1}, \mathrm{~d} x=$ $-\frac{2 t \mathrm{~d} t}{\left(t^{2}-1\right)^{2}}$, 从而所求积分为

$$
\begin{aligned}
\int \frac{1}{x} \sqrt{\frac{1+x}{x}} \mathrm{~d} x & =\int\left(t^{2}-1\right) t \cdot \frac{-2 t}{\left(t^{2}-1\right)^{2}} \mathrm{~d} t=-2 \int \frac{t^{2}}{t^{2}-1} \mathrm{~d} t \\
& =-2 \int\left(1+\frac{1}{t^{2}-1}\right) \mathrm{d} t=-2 t-\ln \left|\frac{t-1}{t+1}\right|+C \\
& =-2 t+2 \ln (t+1)-\ln \left|t^{2}-1\right|+C \\
& =-2 \sqrt{\frac{1+x}{x}}+2 \ln \left(\sqrt{\frac{1+x}{x}}+1\right)+\ln |x|+C .
\end{aligned}
$$

以上四个例子表明, 如果被积函数中含有简单根式 $\sqrt[n]{a x+b}$ 或 $\sqrt[n]{\frac{a x+b}{c x+d}}$, 可 以令这个简单根式为 $u$. 由于这样的变换具有反函数, 且反函数是 $u$ 的有理函 数,因此原积分即可化为有理函数的积分.

## 习 题 4-4

求下列不定积分:

1. $\int \frac{x^{3}}{x+3} \mathrm{~d} x$.
2. $\int \frac{x+1}{x^{2}-2 x+5} \mathrm{~d} x$.
3. $\int \frac{3}{x^{3}+1} \mathrm{~d} x$.
4. $\int \frac{x \mathrm{~d} x}{(x+1)(x+2)(x+3)}$.
5. $\int \frac{\mathrm{d} x}{\left(x^{2}+1\right)\left(x^{2}+x\right)}$.
6. $\int \frac{\mathrm{d} x}{\left(x^{3}+1\right)\left(x^{2}+x+1\right)}$.
7. $\int \frac{-x^{2}-2}{\left(x^{2}+x+1\right)^{2}} \mathrm{~d} x$.
8. $\int \frac{\mathrm{d} x}{3+\cos x}$.
9. $\int \frac{d x}{1+\sin x+\cos x}$.
10. $\int \frac{\mathrm{d} x}{1+\sqrt[3]{x+1}}$.
11. $\int \frac{\sqrt{x+1}-1}{\sqrt{x+1}+1} \mathrm{~d} x$.
12. $\int \sqrt{\frac{1-x}{1+x}} \frac{\mathrm{d} x}{x}$.
13. $\int \frac{2 x+3}{x^{2}+3 x-10} \mathrm{~d} x$.
14. $\int \frac{\mathrm{d} x}{x\left(x^{2}+1\right)}$.
15. $\int \frac{x^{2}+1}{(x+1)^{2}(x-1)} \mathrm{d} x$.
16. $\int \frac{x^{5}+x^{4}-8}{x^{3}-x} \mathrm{~d} x$.
17. $\int \frac{1}{x^{4}-1} \mathrm{~d} x$.
18. $\int \frac{(x+1)^{2}}{\left(x^{2}+1\right)^{2}} \mathrm{~d} x$.
19. $\int \frac{\mathrm{d} x}{3+\sin ^{2} x}$.
20. $\int \frac{d x}{2+\sin x}$.
21. $\int \frac{d x}{2 \sin x-\cos x+5}$.
22. $\int \frac{(\sqrt{x})^{3}-1}{\sqrt{x}+1} \mathrm{~d} x$.
23. $\int \frac{\mathrm{d} x}{\sqrt{x}+\sqrt[4]{x}}$.
24. $\int \frac{\mathrm{d} x}{\sqrt[3]{(x+1)^{2}(x-1)^{4}}}$.

通过前面的讨论可以看出,积分的计算要比导数的计算来得灵活、复杂. 为 了实用的方便, 往往把常用的积分公式汇集成表, 这种表叫做积分表. 积分表是 按照被积函数的类型来排列的. 求积分时, 可根据被积函数的类型直接地或经过 简单的变形后,在表内查得所需的结果.

本书末附录 III 有一个简单的积分表, 以供查阅.

我们先举几个可以直接从积分表中查得结果的积分例子.

例 1 求 $\int \frac{x}{(3 x+4)^{2}} \mathrm{~d} x$. 解 被积函数含有 $a x+b$,在积分表(一)中查得公式(7)

$$
\int \frac{x}{(a x+b)^{2}} \mathrm{~d} x=\frac{1}{a^{2}}\left(\ln |a x+b|+\frac{b}{a x+b}\right)+C .
$$

现在 $a=3 、 b=4$, 于是

$$
\int \frac{x}{(3 x+4)^{2}} \mathrm{~d} x=\frac{1}{9}\left(\ln |3 x+4|+\frac{4}{3 x+4}\right)+C .
$$

例 2 求 $\int \frac{\mathrm{d} x}{5-4 \cos x}$.

解 被积函数含有三角函数,在积分表 (十一) 中查得关于积分 $\int \frac{\mathrm{d} x}{a+b \cos x}$ 的公式,但是公式有两个,要看 $a^{2}>b^{2}$ 或 $a^{2}<b^{2}$ 而决定采用哪一个.

现在 $a=5 、 b=-4, a^{2}>b^{2}$, 所以用公式(105)

$$
\begin{aligned}
& \int \frac{\mathrm{d} x}{a+b \cos x} \\
& =\frac{2}{a+b} \sqrt{\frac{a+b}{a-b}} \arctan \left(\sqrt{\frac{a-b}{a+b}} \tan \frac{x}{2}\right)+C \quad\left(a^{2}>b^{2}\right) .
\end{aligned}
$$

于是

$$
\begin{aligned}
& \int \frac{\mathrm{d} x}{5-4 \cos x} \\
& =\frac{2}{5+(-4)} \sqrt{\frac{5+(-4)}{5-(-4)}} \arctan \left(\sqrt{\frac{5-(-4)}{5+(-4)}} \tan \frac{x}{2}\right)+C \\
& =\frac{2}{3} \arctan \left(3 \tan \frac{x}{2}\right)+C .
\end{aligned}
$$

下面再举一个需要先进行变量代换, 然后再查表求积分的例子.

例 3 求 $\int \frac{\mathrm{d} x}{x \sqrt{4 x^{2}+9}}$.

解 这个积分不能在表中直接查到,需要先进行变量代换.

令 $2 x=u$, 那么 $\sqrt{4 x^{2}+9}=\sqrt{u^{2}+3^{2}}, x=\frac{u}{2}, \mathrm{~d} x=\frac{1}{2} \mathrm{~d} u$.于是

$$
\int \frac{\mathrm{d} x}{x \sqrt{4 x^{2}+9}}=\int \frac{\frac{1}{2} \mathrm{~d} u}{\frac{u}{2} \sqrt{u^{2}+3^{2}}}=\int \frac{\mathrm{d} u}{u \sqrt{u^{2}+3^{2}}} .
$$

被积函数中含有 $\sqrt{u^{2}+3^{2}}$, 在积分表(六) 中查到公式(37)

$$
\int \frac{\mathrm{d} x}{x \sqrt{x^{2}+a^{2}}}=\frac{1}{a} \ln \frac{\sqrt{x^{2}+a^{2}}-a}{|x|}+C .
$$

现在 $a=3, x$ 相当于 $u$,于是

$$
\int \frac{\mathrm{d} u}{u \sqrt{u^{2}+3^{2}}}=\frac{1}{3} \ln \frac{\sqrt{u^{2}+3^{2}}-3}{|u|}+C .
$$

再把 $u=2 x$ 代入, 最后得到

$$
\begin{aligned}
\int \frac{\mathrm{d} x}{x \sqrt{4 x^{2}+9}} & =\int \frac{\mathrm{d} u}{u \sqrt{u^{2}+3^{2}}}=\frac{1}{3} \ln \frac{\sqrt{u^{2}+3^{2}}-3}{|u|}+C . \\
& =\frac{1}{3} \ln \frac{\sqrt{4 x^{2}+9}-3}{2|x|}+C .
\end{aligned}
$$

最后,举一个用递推公式求积分的例子.

例 4 求 $\int \sin ^{4} x \mathrm{~d} x$.

解 在积分表(十一)中查到公式(95)

$$
\int \sin ^{n} x \mathrm{~d} x=-\frac{\sin ^{n-1} x \cos x}{n}+\frac{n-1}{n} \int \sin ^{n-2} x \mathrm{~d} x .
$$

利用这个公式可以使被积函数中正弦的幂次减少两次, 只要重复使用这个 公式,可以使正弦的算次继续减少,直到求出最后结果为止,这种公式叫做递推 公式.

现在 $n=4$, 于是

$$
\int \sin ^{4} x \mathrm{~d} x=-\frac{\sin ^{3} x \cos x}{4}+\frac{3}{4} \int \sin ^{2} x \mathrm{~d} x .
$$

对积分 $\int \sin ^{2} x \mathrm{~d} x$ 用公式(93)

$$
\int \sin ^{2} x \mathrm{~d} x=\frac{x}{2}-\frac{1}{4} \sin 2 x+C,
$$

从而所求积分为

$$
\int \sin ^{4} x \mathrm{~d} x=-\frac{\sin ^{3} x \cos x}{4}+\frac{3}{4}\left(\frac{x}{2}-\frac{1}{4} \sin 2 x\right)+C .
$$

一般说来,查积分表可以节省计算积分的时间,但是, 只有掌握了前面学过 的基本积分方法才能灵活地使用积分表, 而且对一些比较简单的积分, 应用基本 积分方法来计算比査表更快些, 例如, 对 $\int \sin ^{2} x \cos ^{3} x \mathrm{~d} x$, 用变换 $u=\sin x$ 很块

- 就可得到结果. 所以, 求积分时究竟是直接计算, 还是查表, 或是两者结合使用, 应该做具体分析,不能一概而论.

在本章结束之前,我们还要指出:对初等函数来说,在其定义区间上,它的原 函数一定存在,但原函数不一定都是初等函数,如

$$
\int \mathrm{e}^{-x^{2}} \mathrm{~d} x, \quad \int \frac{\sin x}{x} \mathrm{~d} x, \quad \int \frac{\mathrm{d} x}{\ln x}, \quad \int \frac{\mathrm{d} x}{\sqrt{1+x^{4}}}
$$

等等,它们的原函数就都不是初等函数. 利用积分表计算下列不定积分:

1. $\int \frac{\mathrm{d} x}{\sqrt{4 x^{2}-9}}$.
2. $\int \frac{1}{x^{2}+2 x+5} d x$.
3. $\int \frac{\mathrm{d} x}{\sqrt{5-4 x+x^{2}}}$.
4. $\int \sqrt{2 x^{2}+9} \mathrm{~d} x$.
5. $\int \sqrt{3 x^{2}-2} \mathrm{~d} x$.
6. $\int \mathrm{e}^{2 x} \cos x \mathrm{~d} x$.
7. $\int x \arcsin \frac{x}{2} \mathrm{~d} x$.
8. $\int \frac{\mathrm{d} x}{\left(x^{2}+9\right)^{2}}$.
9. $\int \frac{\mathrm{d} x}{\sin ^{3} x}$.
10. $\int \mathrm{e}^{-2 x} \sin 3 x \mathrm{~d} x$.
11. $\int \sin 3 x \sin 5 x \mathrm{~d} x$.
12. $\int \ln ^{3} x \mathrm{~d} x$.
13. $\int \frac{1}{x^{2}(1-x)} \mathrm{d} x$.
14. $\int \frac{\sqrt{x-1}}{x} \mathrm{~d} x$.
15. $\int \frac{1}{\left(1+x^{2}\right)^{2}} \mathrm{~d} x$.
16. $\int \frac{1}{x \sqrt{x^{2}-1}} \mathrm{~d} x$.
17. $\int \frac{x}{(2+3 x)^{2}} \mathrm{~d} x$.
18. $\int \cos ^{6} x \mathrm{~d} x$.
19. $\int x^{2} \sqrt{x^{2}-2} \mathrm{~d} x$.
20. $\int \frac{1}{2+5 \cos x} \mathrm{~d} x$.
21. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{2 x-1}}$.
22. $\int \sqrt{\frac{1-x}{1+x}} \mathrm{~d} x$.
23. $\int \frac{x+5}{x^{2}-2 x-1} \mathrm{~d} x$.
24. $\int \frac{x \mathrm{~d} x}{\sqrt{1+x-x^{2}}}$.
25. $\int \frac{x^{4}}{25+4 x^{2}} \mathrm{~d} x$.

## 总习题 四

求下列不定积分 (其中 $a 、 b$ 为常数):

1. $\int \frac{\mathrm{d} x}{\mathrm{e}^{x}-\mathrm{e}^{-x}}$.
2. $\int \frac{x}{(1-x)^{3}} \mathrm{~d} x$.
3. $\int \frac{x^{2}}{a^{6}-x^{6}} \mathrm{~d} x(a>0)$.
4. $\int \frac{1+\cos x}{x+\sin x} \mathrm{~d} x$.
5. $\int \frac{\ln \ln x}{x} \mathrm{~d} x$.
6. $\int \frac{\sin x \cos x}{1+\sin ^{4} x} \mathrm{~d} x$.
7. $\int \tan ^{4} x \mathrm{~d} x$.
8. $\int \sin x \sin 2 x \sin 3 x \mathrm{~d} x$. 9. $\int \frac{\mathrm{d} x}{x\left(x^{6}+4\right)}$.
9. $\int \frac{\mathrm{d} x}{\sqrt{x(1+x)}}$.
10. $\int \mathrm{e}^{a x} \cos b x \mathrm{~d} x$.
11. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{x^{2}-1}}$.
12. $\int \frac{\mathrm{d} x}{x^{4} \sqrt{1+x^{2}}}$.
13. $\int \ln \left(1+x^{2}\right) \mathrm{d} x$.
14. $\int \arctan \sqrt{x} \mathrm{~d} x$.
15. $\int \frac{x^{3}}{\left(1+x^{8}\right)^{2}} \mathrm{~d} x$.
16. $\int \frac{\mathrm{d} x}{16-x^{4}}$.
17. $\int \frac{x+\sin x}{1+\cos x} \mathrm{~d} x$.
18. $\int \frac{\sqrt[3]{x}}{x(\sqrt{x}+\sqrt[3]{x})} \mathrm{d} x$.
19. $\int \frac{\mathrm{e}^{3 x}+\mathrm{e}^{x}}{\mathrm{e}^{3 x}-\mathrm{e}^{2 x}+1} \mathrm{~d} x$.
20. $\int \ln ^{2}\left(x+\sqrt{1+x^{2}}\right) \mathrm{d} x$.
21. $\int \sqrt{1-x^{2}} \arcsin x \mathrm{~d} x$.
22. $\int \frac{\cot x}{1+\sin x} \mathrm{~d} x$.
23. $\int \frac{\mathrm{d} x}{(2+\cos x) \sin x}$.
24. $\int \sqrt{\frac{a+x}{a-x}} \mathrm{~d} x(a>0)$.
25. $\int x \cos ^{2} x \mathrm{~d} x$.
26. $\int \frac{\mathrm{d} x}{\sqrt{1+\mathrm{e}^{x}}}$.
27. $\int \frac{\mathrm{d} x}{\left(a^{2}-x^{2}\right)^{3 / 2}}$.
28. $\int \sqrt{x} \sin \sqrt{x} \mathrm{~d} x$.
29. $\int \frac{\sin ^{2} x}{\cos ^{3} x} \mathrm{~d} x$.
30. $\int \frac{\sqrt{1+\cos x}}{\sin x} \mathrm{~d} x$.
31. $\int \frac{x^{11}}{x^{8}+3 x^{4}+2} \mathrm{~d} x$.
32. $\int \frac{\sin x}{1+\sin x} \mathrm{~d} x$.
33. $\int \mathrm{e}^{\sin x} \frac{x \cos ^{3} x-\sin x}{\cos ^{2} x} \mathrm{~d} x$.
34. $\int \frac{\mathrm{d} x}{\left(1+\mathrm{e}^{x}\right)^{2}}$.
35. $\int \frac{x \mathrm{e}^{x}}{\left(\mathrm{e}^{x}+1\right)^{2}} \mathrm{~d} x$.
36. $\int \frac{\ln x}{\left(1+x^{2}\right)^{\frac{3}{2}}} \mathrm{~d} x$.
37. $\int \frac{x^{3} \arccos x}{\sqrt{1-x^{2}}} \mathrm{~d} x$.
38. $\int \frac{\mathrm{d} x}{\sin ^{3} x \cos x}$.
39. $\int \frac{\sin x \cos x}{\sin x+\cos x} \mathrm{~d} x$.

## 第五章 定 积 分

本章讨论积分学的另一个基本问题一一定积分问题. 我们先从几何与力学 问题出发引进定积分的定义，然后讨论它的性质与计算方法. 关于定积分的应 用,将在第六章讨论.

## 第一节 定积分的概念与性质

## 一、定积分问题举例

## 1. 曲边梯形的面积

设 $y=f(x)$ 在区间 $[a, b]$ 上非负、连续. 由直线 $x=a 、 x=b 、 y=0$ 及曲线 $y=f(x)$ 所围成的图形 (如图 5-1) 称为典边 梯形,其中曲线弧称为典边.

我们知道,矩形的高是不变的, 它的面积 可按公式

矩形面积 $=$ 高 $\times$ 底

来定义和计算. 而曲边梯形在底边上各点处 的高 $f(x)$ 在区间 $[a, b]$ 上是变动的,故它的 面积不能直接按上述公式来定义和计算. 然

在区间 $[a, b]$ 中任意插入若干个分点

$$
a=x_{0}<x_{1}<x_{2}<\cdots<x_{n-1}<x_{n}=b,
$$

把 $[a, b]$ 分成 $n$ 个小区间

$$
\left[x_{0}, x_{1}\right],\left[x_{1}, x_{2}\right], \cdots,\left[x_{n-1}, x_{n}\right],
$$

它们的长度依次为

$$
\Delta x_{1}=x_{1}-x_{0}, \Delta x_{2}=x_{2}-x_{1}, \cdots, \Delta x_{n}=x_{n}-x_{n-1} .
$$

经过每一个分点作平行于 $y$ 轴的直线段,把曲边梯形分成 $n$ 个窄曲边梯 形. 在每个小区间 $\left[x_{i-1}, x_{i}\right]$ 上任取一点 $\xi_{i}$, 以 $\left[x_{i-1}, x_{i}\right]$ 为底、 $f\left(\xi_{i}\right)$ 为高的窄矩 形近似替代第 $i$ 个察曲边梯形 $(i=1,2, \cdots, n)$ ，把这样得到的 $n$ 个窄矩形面积 之和作为所求曲边梯形面积 $A$ 的近似值, 即

$$
\begin{aligned}
A & \approx f\left(\xi_{1}\right) \Delta x_{1}+f\left(\xi_{2}\right) \Delta x_{2}+\cdots+f\left(\xi_{11}\right) \Delta x_{n} \\
& =\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i} .
\end{aligned}
$$

为了保证所有小区间的长度都无限缩小,我们要求小区间长度中的最大值 趋于零, 如记 $\lambda=\max \left|\Delta x_{1}, \Delta x_{2}, \cdots, \Delta x_{n}\right|$, 则上述条件可表为 $\lambda \rightarrow 0$. 当 $\lambda \rightarrow 0$ 时 (这时分段数 $n$ 无限增多, 即 $n \rightarrow \infty$ ), 取上述和式的极限, 便得曲边梯形的面积

$$
A=\lim _{i \rightarrow 0} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i} .
$$

## 2. 变速直线运动的路程

设某物体作直线运动, 已知速度 $v=v(t)$ 是时间间隔 $\left[T_{1}, T_{2}\right]$ 上 $t$ 的连续 函数, 且 $v(t) \geqslant 0$,计算在这段时间内物体所经过的路程 $s$.

我们知道, 对于等速直线运动, 有公式

路程 $=$ 速度 $\times$ 时间.

但是, 在现在讨论的问题中；速度不是常量而是随时间变化的变量，因此，所求路 程 $s$ 不能直接按等速直线运动的路程公式来计算. 然而,物体运动的速度函数 $v=v(t)$ 是连续变化的, 在很短一段时间内, 速度的变化很小, 近似于等速. 因 此,如果把时间间隔分小,在小段时间内,以等速运动代替变速运动, 那么; 就可 算出部分路程的近似值;再求和，得到整个路程的近似值;最后，通过对时间间隔 无限细分的极限过程,这时所有部分路程的近似值之和的极限, 就是所求变速直 线运动的路程的精确值.

具体计算步骤如下:

在时间间隔 $\left[T_{1}, T_{2}\right.$ ]内任意插入若干个分点

$$
T_{1}=t_{0}<t_{1}<t_{2}<\cdots<t_{n-1}<t_{n}=T_{2},
$$

把 $\left[T_{1}, T_{2}\right]$ 分成 $n$ 个小时段

$$
\left[t_{0}, t_{1}\right],\left[t_{1}, t_{2}\right], \cdots,\left[t_{n-1}, t_{n}\right],
$$

各小时段时间的长依次为

$$
\Delta t_{1}=t_{1}-t_{0}, \Delta t_{2}=t_{2}-t_{1}, \cdots, \Delta t_{n}=t_{n}-t_{n-1} .
$$

相应的;在各段时间内物体经过的路程依次为

$$
\Delta s_{1}, \Delta s_{2}, \cdots, \Delta s_{n} .
$$

在时间间隔 $\left[t_{i-1}, t_{i}\right]$ 上任取一个时刻 $\tau_{i}\left(t_{i-1} \leqslant \tau_{i} \leqslant t_{i}\right)$, 以 $\tau_{i}$ 时的速度 $v\left(\tau_{i}\right)$ 来代替 $\left[t_{i-1}, t_{i}\right]$ 上各个时刻的速度, 得到部分路程 $\Delta s_{i}$ 的近似值, 即

$$
\Delta s_{i} \approx v\left(\tau_{i}\right) \Delta t_{i} \quad(i=1,2, \cdots, n) .
$$

于是这 $n$ 段部分路程的近似值之和就是所求变速直线运动路程 $s$ 的近似值, 即

$$
\begin{aligned}
s & \approx v\left(\tau_{1}\right) \Delta t_{1}+v\left(\tau_{2}\right) \Delta t_{2}+\cdots+v\left(\tau_{n}\right) \Delta t_{n} \\
& =\sum_{i=1}^{n} v\left(\tau_{i}\right) \Delta t_{i} .
\end{aligned}
$$

记 $\lambda=\max \left\{\Delta t_{1}, \Delta t_{2}, \cdots, \Delta t_{n}\right\}$, 当 $\lambda \rightarrow 0$ 时, 取上述和式的极限, 即得变速直 线运动的路程

$$
s=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} v\left(\tau_{i}\right) \Delta t_{i}
$$

## 二、定积分定义

从上面两个例子可以看到:所要计算的量,即曲边梯形的面积 $A$ 及变速直 线运动的路程 $s$ 的实际意义虽然不同,前者是几何量,后者是物理量,但是它们 都决定于一个函数及其自变量的变化区间,如:

曲边梯形的高度 $y=f(x)$ 及其底边上的点 $x$ 的变化区间 $[a, b]$ ，

直线运动的速度 $v=v(t)$ 及时间 $t$ 的变化区间 $\left[T_{1}, T_{2}\right]$;

其次, 计算这些量的方法与步骤都是相同的, 并且它们都归结为具有相同结 构的一种特定和的极限, 如

$$
\begin{aligned}
& \text { 面积 } A=\lim _{\lambda-01} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}, \\
& \text { 路程 } s=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} v\left(\tau_{i}\right) \Delta t_{i} .
\end{aligned}
$$

拖开这些问题的具体意义,抓住它们在数量关系上共同的本质与特性加以 概括,我们就可以抽象出下述定积分的定义.

定义 设函数 $f(x)$ 在 $[a, b]$ 上有界, 在 $[a, b]$ 中任意插入苦干个分点

$$
a=x_{0}<x_{1}<x_{2}<\cdots<x_{n-1}<x_{n}=b,
$$

把区间 $[a, b]$ 分成 $n$ 个小区间

$$
\left[x_{0}, x_{1}\right],\left[x_{1}, x_{2}\right], \cdots,\left[x_{n-1}, x_{n}\right],
$$

## 各个小区间的长度依次为

$$
\Delta x_{1}=x_{1}-x_{1}, \Delta x_{2}=x_{2}-x_{1}, \cdots, \Delta x_{n}=x_{n}-x_{n-1} \text { 。 }
$$

在每个小区间 $\left[x_{i-1}, x_{i}\right]$ 上任取一点 $\xi_{i} \quad\left(x_{i-1} \leqslant \xi_{i} \leqslant x_{i}\right)$, 作函数值 $f\left(\xi_{i}\right)$ 与小 区间长度 $\Delta x_{i}$ 的乘积 $f\left(\xi_{i}\right) \Delta x_{i} \quad(i=1,2, \cdots, n)$, 并作出和

$$
S=\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i} .
$$

记 $\lambda=\max \left\{\Delta x_{1}, \Delta x_{2}, \cdots, \Delta x_{n}\right\}$ ，如果不论对 $[a, b]$ 怎样划分，也不论在小区 间 $\left[x_{i-1}, x_{i}\right]$ 上点 $\xi_{i}$ 怎样选取, 只要当 $\lambda \rightarrow 0$ 时, 和 $S$ 总趋于确定的极限 $I$, 那么 称这个极限 $I$ 为函数 $f(x)$ 在区间 $[a, b]$ 上的定积分 (简称积分), 记作 $\int_{a}^{b} f(x) \mathrm{d} x$, 即

$$
\text { , } \quad \int_{a}^{b} f(x) \mathrm{d} x=I=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i},
$$

其中 $f(x)$ 叫做被积函数, $f(x) \mathrm{d} x$ 叫做被积表达式, $x$ 叫做积分变量, $a$ 叫做积 分下限, $b$ 叫做积分上限, $[a, b]$ 叫做积分区间.

利用 “ $\varepsilon-\delta$ ”的说法, 上述定积分的定义可以表述如下:

设有常数 $I$,如果对于任意给定的正数 $\varepsilon$, 总存在一个正数 $\delta$,使得对于区间 $[a, b]$ 的任何分法,不论 $\xi_{i}$ 在 $\left[x_{i-1}, x_{i}\right]$ 中怎样选取, 只要 $\lambda<\delta$, 总有

$$
\left|\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}-I\right|<\varepsilon
$$

成立,则称 $I$ 是 $f(x)$ 在区间 $[a, b]$ 上的定积分, 记作 $\int_{a}^{b} f(x) \mathrm{d} x$.

注意 当和 $\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}$ 的极限存在时, 其极限 $I$ 仅与被积函数 $f(x)$ 及 积分区间 $[a, b]$ 有关，如果既不改变被积函数 $f$, 也不改变积分区间 $[a, b]$, 而只 把积分变量 $x$ 改写成其他字母,例如 $t$ 或 $u$, 那么, 这时和的极限 $I$ 不变,也就是 定积分的值不变, 即

$$
\int_{a}^{b} f(x) \mathrm{d} x=\int_{a}^{b} f(t) \mathrm{d} t=\int_{u}^{b} f(u) \mathrm{d} u .
$$

这就是说,定积分的值只与被积函数及积分区间有关,而与积分变量的记法无 关.

和 $\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}$ 通常称为 $f(x)$ 的积分和. 如果 $f(x)$ 在 $[a, b]$ 上的定积分 存在,那么就说 $f(x)$ 在 $[a, b]$ 上可积.

对于定积分, 有这样一个重要问题: 函数 $f^{\prime}(x)$ 在 $[a, b]$ 上满足怎样的条件, $f(x)$ 在 $[a, b]$ 上一定可积? 这个问题我们不作深人讨论, 而只给出以下两个充 分条件.

定理 1 设 $f(x)$ 在区间 $[a, b]$ 上连续,则 $f(x)$ 在 $[a, b]$ 上可积. 定理 2 设 $f(x)$ 在区间 $[a, b]$ 上有界, 且只有有限个间断点, 则 $f(x)$ 在 $[a, b]$ 上可积.

利用定积分的定义,前面所讨论的两个实际问题可以分别表述如下:

曲线 $y=f(x)(f(x) \geqslant 0) 、 x$ 轴及两条直线 $x=a 、 x=b$ 所围成的曲边梯 形的面积 $A$ 等于函数 $f(x)$ 在区间 $[a, b]$ 上的定积分. 即

$$
A=\int_{a}^{b} f(x) \mathrm{d} x .
$$

物体以变速 $v=v(t)(v(t) \geqslant 0)$ 作直线运动, 从时刻 $t=T_{1}$ 到时刻 $t=$ $T_{2}$, 这物体经过的路程 $s$ 等于函数 $v(\iota)$ 在区间 $\left[T_{1}, T_{2}\right]$ 上的定积分, 即

$$
s=\int_{T_{1}}^{T_{2}} v(t) \mathrm{d} t .
$$

下面讨论定积分的几何意义. 在 $[a, b]$ 上 $f(x) \geqslant 0$ 时,我们已经知道,定积 分 $\int_{a}^{b} f(x) \mathrm{d} x$ 在几何上表示由曲线 $y=f(x)$ 、两条直线 $x=a 、 x=b$ 与 $x$ 轴所围 成的曲边梯形的面积; 在 $[a, b]$ 上 $f(x) \leqslant 0$ 时, 由曲线 $y=f(x)$ 、两条直线 $x=a 、 x=b$ 与 $x$ 轴所围成的曲边梯形位于 $x$ 轴的下方, 定积分

$$
\int_{a}^{b} f(x) \mathrm{d} x
$$

在几何上表示上述曲边梯形面积的负值; 在 $[a, b]$ 上 $f(x)$ 既取得正值又取得负值时, 函 数 $f(x)$ 的图形某些部分在 $x$ 轴的上方,而其 他部分在 $x$ 轴下方 (图 5-2), 此时定积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 表示 $x$ 轴上方图形面积减去 $x$ 轴下 方图形面积所得之差.

最后,举一个按定义计算定积分的例子.

例 1 利用定义计算定积分 $\int_{0}^{1} x^{2} \mathrm{~d} x$.

解 因为被积函数 $f(x)=x^{2}$ 在积分区间 $[0,1]$ 上连续,而连续函数是可积 的,所以积分与区间 $[0,1]$ 的分法及点 $\xi_{i}$ 的取法无关. 因此, 为了便于计算,不妨 把区间 $[0,1]$ 分成 $n$ 等份, 分点为 $x_{i}=\frac{i}{n}, i=1,2, \cdots, n-1$; 这样, 每个小区间 $\left[x_{i-1}, x_{i}\right]$ 的长度 $\Delta x_{i}=\frac{1}{n}, i=1,2, \cdots, n$; 取 $\xi_{i}=x_{i}, i=1,2, \cdots, n$.于是, 得和 式

$$
\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}=\sum_{i=1}^{n} \xi_{i}^{2} \Delta x_{i}=\sum_{i=1}^{n} x_{i}^{2} \Delta x_{i}
$$

$$
\begin{aligned}
& =\sum_{i=1}^{n}\left(\frac{i}{n}\right)^{2} \cdot \frac{1}{n}=\frac{1}{n^{3}} \sum_{i=1}^{n} i^{2} \\
& =\frac{1}{n^{3}} \cdot \frac{1}{6} n(n+1)(2 n+1)(1) \\
& =\frac{1}{6}\left(1+\frac{1}{n}\right)\left(2+\frac{1}{n}\right) .
\end{aligned}
$$

当 $\lambda \rightarrow 0$ 即 $n \rightarrow \infty$ 时, 取上式右端的极限. 由定积分的定义, 即得所要计算的 积分为

$$
\int_{0}^{1} x^{2} \mathrm{~d} x=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} \xi_{i}^{2} \Delta x_{i}=\lim _{n \rightarrow \infty} \frac{1}{6}\left(1+\frac{1}{n}\right)\left(2+\frac{1}{n}\right)=\frac{1}{3} .
$$

## 三、定积分的近似计算

：从例 1 的计算过程中可以看到,对于任一确定的正整数 $n$, 积分和

$$
\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}=\frac{1}{6}\left(1+\frac{1}{n}\right)\left(2+\frac{1}{n}\right)
$$

都是定积分 $\int_{0}^{1} x^{2} \mathrm{~d} x$ 的近似值. 当 $n$ 取不同值时, 可得到定积分 $\int_{0}^{1} x^{2} \mathrm{~d} x$ 精度 不同的近似值.一般说来, $n$ 取得越大,近似程度越好.

下面就一般情形,讨论定积分的近似计算问题. 设 $f(x)$ 在 $[a, b]$ 上连续,这 时定积分 $\int_{0}^{b} f(x) \mathrm{d} x$ 存在. 如同例 1 , 采取把区间 $[a, b]$ 等分的分法, 即用分点 $a=x_{0}, x_{1}, x_{2}, \cdots, x_{n}=b$ 将 $[a, b]$ 分成 $n$ 个长度相等的小区间, 每个小区间的 长为

(1) 利用桓等式 $(n+1)^{3}=n^{3}+3 n^{2}+3 n+1$, 得

$$
\left\{\begin{array}{l}
(n+1)^{3}-n^{3}=3 n^{2}+3 n+1 \\
n^{3}-(n-1)^{3}=3(n-1)^{2}+3(n-1)+1 \\
\cdots \ldots \ldots \ldots \\
3^{3}-2^{3}=3 \cdot 2^{2}+3 \cdot 2+1 \\
2^{3}-1^{3}=3 \cdot 1^{2}+3 \cdot 1+1
\end{array}\right.
$$

把这 $n$ 个等式两端分别相加，得

$$
(n+1)^{3}-1=3\left(1^{2}+2^{2}+\cdots+n^{2}\right)+3(1+2+\cdots+n)+n .
$$

中于

$$
1+2+\cdots+n=\frac{1}{2} n(n+1) \text {. }
$$

代入上式,得

$$
n^{3}+3 n^{2}+3 n=3\left(1^{2}+2^{2}+\cdots+n^{2}\right)+\frac{3}{2} n(n+1)+n .
$$

整理后, 得

$$
1^{2}+2^{2}+\cdots+n^{2}=\frac{1}{6} n(n+1)(2 n+1) .
$$

$$
\Delta x=\frac{b-a}{n},
$$

在小区间 $\left[x_{i-1}, x_{i}\right]$ 上, 取 $\xi_{i}=x_{i-1}$, 应有

$$
\int_{a}^{b} f(x) \mathrm{d} x=\lim _{n \rightarrow \infty} \frac{b-a}{n} \sum_{i=1}^{n} f\left(x_{i-1}\right),
$$

从而对于任一确定的正整数 $n$, 有

$$
\int_{a}^{b} f(x) \mathrm{d} x \approx \frac{b-a}{n} \sum_{i=1}^{n} f\left(x_{i-1}\right) .
$$

记 $f\left(x_{i}\right)=y_{i} \quad(i=0,1,2, \cdots, n)$, 上式可记作

$$
\int_{a}^{l} f(x) \mathrm{d} x \approx \frac{b-a}{n}\left(y_{0}+y_{1}+\cdots+y_{n-1}\right) .
$$

如果取 $\xi_{i}=x_{i}$, 则可得近似公式

$$
\int_{a}^{b} f(x) \mathrm{d} x \approx \frac{b-a}{n}\left(y_{1}+y_{2}+\cdots+y_{n}\right) .
$$

以上求定积分近似值的方法称为䓡形法, 公式 (3)、(4)称为矩形法公式.

矩形法的几何意义是: 用窄条矩形的面积作 为窄条曲边梯形面积的近似值. 整体上用台阶形 的面积作为曲边梯形面积的近似值. 如图 5-3 所示.

求定积分近似值的方法, 常用的还有梯形法 和抛物线法 (又称辛普森 (Simpson) 法), 简单介 绍如下.

和矩形法一样, 将区间 $[a, b] n$ 等分. 设 $f\left(x_{i}\right)=y_{i}$, 曲线 $y=f(x)$ 上的点 $\left(x_{i}, y_{i}\right)$ 记作

$M_{i}(i=0,1,2, \cdots, n)$.

梯形法的原理是: 将曲线 $y=f(x)$ 上的小弧段 $\widehat{M}_{i-1} M_{i}$ 用直线段 $\bar{M}_{i-1} M_{i}$ 代 替, 也就是把窄条曲边梯形用窄条梯形代替 (图 5-4(a)), 由此得到定积分的近 似值为

$$
\begin{aligned}
\int_{a}^{b} f(x) \mathrm{d} x & \approx \frac{b-a}{n}\left(\frac{y_{0}+y_{1}}{2}+\frac{y_{1}+y_{2}}{2}+\cdots+\frac{y_{n-1}+y_{n}}{2}\right) \\
& =\frac{b-a}{n}\left(\frac{y_{0}+y_{n}}{2}+y_{1}+y_{2}+\cdots+y_{n-1}\right) .
\end{aligned}
$$

显然, 梯形法公式 (5) 所得近似值就是矩形法公式 (3) 和 (4) 所得两个近似值的平 均值.

拖物线法的原理是: 将曲线 $y=f(x)$ 上的两个小弧段 $\widehat{M}_{i-1} M_{i}$ 和 ${\widehat{M}, M_{i+1}}_{1}$ 合

## 因 5-4

起来, 用过 $M_{i-1}, M_{i}, M_{i+1}$ 三点的抛物线 $y=p x^{2}+q x+r$ 代替 (图 5-4(b)). 经推导可得, 以此抛物线弧段为曲边、以 $\left[x_{i-1}, x_{i+1}\right]$ 为底的曲边梯形面积为

$$
\frac{1}{6}\left(y_{i-1}+4 y_{i}+y_{i+1}\right) \cdot 2 \Delta x=\frac{b-a}{3 n}\left(y_{i-1}+4 y_{i}+y_{i+1}\right) \text {. }
$$

取 $n$ 为偶数,得到定积分的近似值为

$$
\begin{aligned}
\int_{a}^{b} f(x) \mathrm{d} x & \approx \frac{b-a}{3 n}\left[\left(y_{0}+4 y_{1}+y_{2}\right)+\left(y_{2}+4 y_{3}+y_{4}\right)+\cdots+\left(y_{n-2}+4 y_{n-1}+y_{n}\right)\right] \\
& =\frac{b-a}{3 n}\left[y_{0}+y_{n}+4\left(y_{1}+y_{3}+\cdots+y_{n-1}\right)+2\left(y_{2}+y_{4}+\cdots+y_{n-2}\right)\right] .
\end{aligned}
$$

例 2 按梯形法公式 (5) 和抛物线法公式 (6) 计算定积分 $\int_{0}^{1} \frac{4}{1+x^{2}} \mathrm{~d} x$ 的近 似值. (取 $n=10$, 计算时取 5 位小数).

解 计算 $y_{i}$ 并列表:

| $i$ | $x_{i}$ | $y_{i}$ |
| :---: | :---: | :---: |
| 0 | 0.0 | 4.00000 |
| 1 | 0.1 | 3.96040 |
| 2 | 0.2 | 3.84615 |
| 3 | 0.3 | 3.66972 |
| 4 | 0.4 | 3.44828 |
| 5 | 0.5 | 3.20000 |
| 6 | 0.6 | 2.94118 |
| 7 | 0.7 | 2.68456 |
| 8 | 0.8 | 2.43902 |
| 9 | 0.9 | 2.20994 |
| 10 | 1.0 | 2.00000 |

按梯形法公式 (5)求得近似值为

$$
S_{1}=3.13993 ;
$$

按抛物线法公式 (6) 求得近似值为

$$
S_{2}=3.14159 .
$$

本例所给积分的精确值为

$$
\pi=3.1415926 \cdots,
$$

用 $S_{2}$ 作为 $\pi$ 的近似值, 误差小于 $10^{-5}$.

计算定积分的近似值的方法很多,这里不再作介绍. 随着计算机应用的普 及,定积分的近似计算已变得更为方便,现在已有很多现成的数学软件可用于定 积分的近似计算.

## 四、定积分的性质

为了以后计算及应用方便起见, 对定积分作以下两点补充规定：

（1）当 $a=b$ 时, $\int_{a}^{b} f(x) \mathrm{d} x=0$ ；

(2) 当 $a>b$ 时, $\int_{a}^{b} f(x) \mathrm{d} x=-\int_{b}^{a} f(x) \mathrm{d} x$.

由上式可知, 交换定积分的上下限吋,定积分的绝对值不变而符兵相反.

下面讨论定积分的性质.下列各性质中积分上下限的大小, 如不特别指明, 均不加限制 ; 并假定各性质中所列出的定积分都是存在的.

性质 $1 \int_{a}^{b}[f(x) \pm g(x)] \mathrm{d} x=\int_{a}^{b} f(x) \mathrm{d} x \pm \int_{a}^{b} g(x) \mathrm{d} x$.

证 $\int_{a}^{b}[f(x) \pm g(x)] \mathrm{d} x=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n}\left[f\left(\xi_{i}\right) \pm g\left(\xi_{i}\right)\right] \Delta x_{i}$

$$
\begin{aligned}
& =\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i} \pm \lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} g\left(\xi_{i}\right) \Delta x_{i} \\
& =\int_{a}^{\prime \prime} f(x) \mathrm{d} x \pm \int_{a}^{b} g(x) \mathrm{d} x .
\end{aligned}
$$

性质 1 对于任意有限个函数都是成立的. 类似地, 可以证明:

性质 $2 \int_{a}^{b} k f(x) \mathrm{d} x=k \int_{a}^{b} f(x) \mathrm{d} x$ ( $k$ 是常数).

性质 3 设 $a<c<b$, 则

$$
\int_{a}^{b} f(x) \mathrm{d} x=\int_{a}^{c} f(x) \mathrm{d} x+\int_{c}^{b} f(x) \mathrm{d} x .
$$

证 因为函数 $f(x)$ 在区间 $[a, b]$ 上可积,所以不论把 $[a, b]$ 怎样分, 积分 和的极限总是不变的. 因此, 在分区间时, 可以使 $c$ 永远是个分点. 那么, $[a, b]$ 上的积分和等于 $[a, c]$ 上的积分和加 $[c, b]$ 上的积分和, 记为

$$
\sum_{[a, b]} f\left(\xi_{i}\right) \Delta x_{i}=\sum_{[a, c]} f\left(\xi_{i}\right) \Delta x_{i}+\sum_{[a, b]} f\left(\xi_{i}\right) \Delta x_{i} .
$$

令 $\lambda \rightarrow 0$, 上式两端同时取极限, 即得

$$
\int_{a}^{b} f(x) \mathrm{d} x=\int_{a}^{c} f(x) \mathrm{d} x+\int_{c}^{b} f(x) \mathrm{d} x .
$$

这个性质表明定积分对于积分区间具有可加性.

按定积分的补充规定, 我们有: 不论 $a, b, c$ 的相对位置如何, 总有等式

$$
\int_{a}^{h} f(x) \mathrm{d} x=\int_{a}^{c} f(x) \mathrm{d} x+\int_{c}^{b} f(x) \mathrm{d} x
$$

成立. 例如, 当 $a<b<c$ 时, 由于

$$
\int_{a}^{c} f(x) \mathrm{d} x=\int_{a}^{b} f(x) \mathrm{d} x+\int_{b}^{b} f(x) \mathrm{d} x,
$$

于是得

$$
\begin{aligned}
\int_{a}^{b} f(x) \mathrm{d} x & =\int_{a}^{b} f(x) \mathrm{d} x-\int_{b}^{c} f(x) \mathrm{d} x \\
& =\int_{a}^{c} f(x) \mathrm{d} x+\int_{c}^{b} f(x) \mathrm{d} x .
\end{aligned}
$$

性质 4 如果在区间 $[a, b]$ 上 $f(x) \equiv 1$, 则

$$
\int_{a}^{b} 1 \mathrm{~d} x=\int_{a}^{b} \mathrm{~d} x=b-a .
$$

这个性质的证明请读者自己完成.

性质 5 如果在区间 $[a, b]$ 上, $f(x) \geqslant 0$, 则

$$
\int_{11}^{b} f(x) \mathrm{d} x \geqslant 0 \quad(a<b) .
$$

证 因为 $f(x) \geqslant 0$, 所以

$$
f\left(\xi_{i}\right) \geqslant 0 \quad(i=1,2, \cdots, n) .
$$

又由于 $\Delta x_{i} \geqslant 0 \quad(i=1,2, \cdots, n)$, 因此

$$
\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i} \geqslant 0,
$$

令 $\lambda=\max \left\{\Delta x_{1}, \cdots, \Delta x_{n}\right\} \rightarrow 0$, 便得要证的不等式.

推论 1 如果在区间 $[a, b]$ 上, $f(x) \leqslant g(x)$, 则

$$
\int_{a}^{b} f(x) \mathrm{d} x \leqslant \int_{a}^{b} g(x) \mathrm{d} x \quad(a<b) .
$$

证 因为 $g(x)-f(x) \geqslant 0$, 由性质 5 得

$$
\int_{a}^{\prime \prime}[g(x)-f(x)] \mathrm{d} x \geqslant 0 .
$$

再利用性质 1 , 便得要证的不等式.

推论 $2\left|\int_{a}^{\prime \prime} f(x) \mathrm{d} x\right| \leqslant \int_{a}^{b}|f(x)| \mathrm{d} x \quad(a<b)$.

证 因为

$$
-|f(x)| \leqslant f(x) \leqslant|f(x)|
$$

所以由推论 1 及性质 2 可得

$$
-\int_{a}^{b}|f(x)| \mathrm{d} x \leqslant \int_{a}^{b} f(x) \mathrm{d} x \leqslant \int_{a}^{b}|f(x)| \mathrm{d} x,
$$

即

$$
\left|\int_{a}^{h} f(x) \mathrm{d} x\right| \leqslant \int_{a}^{b}|f(x)| \mathrm{d} x .
$$

性质 6 设 $M$ 及 $m$ 分别是函数 $f(x)$ 在区间 $[a, b]$ 上的最大值及最小值, 则

$$
m(b-a) \leqslant \int_{a}^{b} f(x) \mathrm{d} x \leqslant M(b-a) \quad(a<b) .
$$

证 因为 $m \leqslant f(x) \leqslant M$, 所以由性质 5 推论 1 , 得

$$
\int_{a}^{b,} m \mathrm{~d} x \leqslant \int_{a}^{b} f(x) \mathrm{d} x \leqslant \int_{a}^{h} M \mathrm{~d} x:
$$

再由性质 2 及性质 4 , 即得所要证的不等式.

这个性质说明, 由被积函数在积分区间上的最大值及最小值, 可以估计积分 值的大致范围. 例如, 定积分 $\int_{\frac{1}{2}}^{1} x^{4} \mathrm{~d} x$, 它的被积函数 $f(x)=x^{4}$ 在积分区间 $\left[\frac{1}{2}, 1\right]$ 上是单调增加的, 于是有最小值 $m=\left(\frac{1}{2}\right)^{4}=\frac{1}{16}$ 、最大值 $M=(1)^{4}=1$. 由性质 6 , 得

$$
\frac{1}{16}\left(1-\frac{1}{2}\right) \leqslant \int_{\frac{1}{2}}^{1} x^{4} \cdot \mathrm{d} x \leqslant 1 \cdot\left(1-\frac{1}{2}\right) \text {, }
$$

即

$$
\frac{1}{32} \leqslant \int_{\frac{1}{2}}^{1} x^{4} \mathrm{~d} x \leqslant \frac{1}{2} .
$$

性质 7(定积分中值定理) 如果函数 $f(x)$ 在积分区间 $[a, b]$ 上连续,则在 $[a, b]$ 上至少存在一个点 $\xi$, 使下式成立:

$$
\int_{a}^{b} f(x) \mathrm{d} x=f(\xi)(b-a) \quad(a \leqslant \xi \leqslant b) .
$$

这个公式叫做积分中值公式.

证 把性质 6 中的不等式各除以 $b-a$, 得

$$
m \leqslant \frac{1}{b-a} \int_{a}^{b} f(x) \mathrm{d} x \leqslant M .
$$

这表明, 确定的数值 $\frac{1}{b-a} \int_{a}^{b} f(x) \mathrm{d} x$ 介于函数 $f(x)$ 的最小值 $m$ 及最大值 $M$ 之 间. 根据闭区间上连续函数的介值定理 (第一章第十节定理 3 推论), 在 $[a, b]$ 上 至少存在一点 $\xi$, 使得函数 $f(x)$ 在点 $\xi$ 处的值与这个确定的数值相等, 即应有

$$
\frac{1}{b-a} \int_{a}^{h} f(x) \mathrm{d} x=f(\xi) \quad(a \leqslant \xi \leqslant b) .
$$

两端各乘以 $b-a$; 即得所要证的等式.

显然, 积分中值公式

$$
\int_{a}^{b} f(x) \mathrm{d} x=f(\xi)(b-a) .(\xi \text { 在 } a \text { 与 } b \text { 之间 })
$$

不论 $a<b$ 或 $a>b$ 都是成立的.

积分中值公式有如下的几何解释: 在区 间 $[a, b]$ 上至少存在一点 $\xi$, 使得以区间 $[a$, $b]$ 为底边、以曲线 $y=f(x)$ 为曲边的曲边梯 形的面积等于同一底边而高为 $f(\xi)$ 的一个 矩形的面积 (图 5-5).

按积分中值公式所得

$$
f(\xi)=\frac{1}{b-a} \int_{a}^{b} f(x) \mathrm{d} x
$$

称为函数 $f(x)$ 在区间 $[a, b]$ 上的平均值. 例 如按图 5-5, $f(\xi)$ 可看作图中曲边梯形的平均高度. 又如物体以变速 $v(t)$ 作直 线运动, 在时间区间 $\left[T_{1}, T_{2}\right]$ 上经过的路程为 $\int_{T_{1}}^{T_{2}} v(t) \mathrm{d} t$, 因此,

$$
v(\xi)=\frac{1}{T_{2}-T_{1}} \int_{r_{1}}^{T_{2}} v(t) \mathrm{d} t, \xi \in\left[T_{1}, T_{2}\right]
$$

便是运动物体在 $\left[T_{1}, T_{2}\right]$ 这段时间内的平均速度.

## 习 题 5-1

1. 利用定积分定义计算由地物线 $y=x^{2}+1$, 两直线 $x=a, x=b \quad(b>a)$ 及 $x$ 轴所围 成的图形的面积.

-2. 利用定积分定义计算下列积分:
(1) $\int_{a}^{b} x \mathrm{~d} x \quad(a<b)$;
(2) $\int_{0}^{1} \mathrm{e}^{x} \mathrm{~d} x$.

3. 利用定积分的几何意义,证明下列等式:
(1) $\int_{0}^{1} 2 x \mathrm{~d} x=1$;
(2) $\int_{0}^{1} \sqrt{1-x^{2}} \mathrm{~d} x=\frac{\pi}{4}$;
(3) $\int_{-\pi}^{\pi} \sin x \mathrm{~d} x=0$;
(4) $\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos x \mathrm{~d} x=2 \int_{0}^{\frac{\pi}{2}} \cos x \mathrm{~d} x$.
4. 利用定积分的几何意义,求下列积分:
(1) $\int_{0}^{t} x \mathrm{~d} x \quad(t>0)$;
(2) $\int_{-2}^{4}\left(\frac{x}{2}+3\right) \mathrm{d} x$;
(3) $\int_{-1}^{2}|x| \mathrm{d} x$;
(4) $\int_{-3}^{3} \sqrt{9-x^{2}} \mathrm{~d} x$.
5. 设 $a<b$. 问 $a 、 b$ 取什么值时，积分 $\int_{a}^{b}\left(x-x^{2}\right) \mathrm{d} x$ 取得最大值?
6. 已知 $\ln 2=\int_{1}^{1} \frac{1}{1+x} \mathrm{~d} x$, 试用抽物线法公式(6), 求出 $\ln 2$ 的近似值(取 $n=10$, 计算时 取 4 位小数).
7. 设 $\int_{-1}^{1} 3 f(x) \mathrm{d} x=18, \int_{-1}^{3} f(x) \mathrm{d} x=4, \int_{-1}^{3} g(x) \mathrm{d} x=3$. 求
(1) $\int_{-1}^{1} f(x) \mathrm{d} x$;
(2) $\int_{1}^{3} f(x) \mathrm{d} x$;
(3) $\int_{3}^{-1} g(x) \mathrm{d} x$;
(4) $\int_{-1}^{3} \frac{1}{5}[4 f(x)+3 g(x)] \mathrm{d} x$.
8. 水利工程中要计算拦水闸门所受的水压力. 已知闸门上水的压强 $p$ 与水深 $h$ 存在函 数关系, 且有 $p=9.8 h\left(\mathrm{kN} / \mathrm{m}^{2}\right)$. 若闸门高 $H=3 \mathrm{~m}$, 宽 $L=2 \mathrm{~m}$, 求水面与闸门顶相齐时闸门 所受的水压力 $P$.

## 9. 证明定积分性质:

(1) $\int_{a}^{b} k f(x) \mathrm{d} x=k \int_{a}^{b} f(x) \mathrm{d} x \quad(k$ 是常数);
(2) $\int_{a}^{b} 1 \cdot \mathrm{d} x=\int_{a}^{b} \mathrm{~d} x=b-a$.

10. 估计下列各积分的值:
(1) $\int_{1}^{4}\left(x^{2}+1\right) \mathrm{d} x$;
(2) $\int_{\frac{\pi}{4}}^{\frac{5}{4} \pi}\left(1+\sin ^{2} x\right) \mathrm{d} x$;
(3) $\int_{\frac{1}{\sqrt{3}}}^{\sqrt{3}} x \arctan x \mathrm{~d} x$;
(4) $\int_{2}^{10} \mathrm{e}^{x^{2}-. x} \mathrm{~d} x$.
11. 设 $f(x)$ 在 $[0,1]$ 上连续, 证明 $\int_{0}^{1} f^{2}(x) \mathrm{d} x \geqslant\left(\int_{0}^{1} f(x) \mathrm{d} x\right)^{2}$.
12. 设 $f(x)$ 及 $g(x)$ 在 $[a, b]$ 上连续,证明: (1) 若在 $[a, b]$ 上, $f(x) \geqslant 0$, 且 $\int_{a}^{b} f(x) \mathrm{d} x=0$, 则在 $[a, b]$ 上 $f(x) \equiv 0$ ；

(2) 若在 $[a, b]$ 上, $f(x) \geqslant 0$, 且 $f(x) \neq 0$, 则 $\int_{a}^{\prime \prime} f(x) \mathrm{d} x>0$;

(3) 若在 $[a, b]$ 上, $f(x) \leqslant g(x)$, 且 $\int_{a}^{b} f(x) \mathrm{d} x=\int_{a}^{b} g(x) \mathrm{d} x$, 则在 $[a, b]$ 上 $f(x) \equiv$ $g(x)$.

13. 根据定积分的性质及第 12 题的结论,说明下列各对积分哪一个的值较大:

(1) $\int_{0}^{1} x^{2} \mathrm{~d} x$ 还是 $\int_{01}^{1} x^{3} \mathrm{~d} x$ ?

(2) $\int_{1}^{2} x^{2} \mathrm{~d} x$ 还是 $\int_{1}^{2} x^{3} \mathrm{~d} x$ ?

(3) $\int_{1}^{2} \ln x \mathrm{~d} x$ 还是 $\int_{1}^{2}(\ln x)^{2} \mathrm{~d} x$ ?

(4) $\int_{0}^{1} x \mathrm{~d} x$ 还是 $\int_{0}^{1} \ln (1+x) \mathrm{d} x$ ?

(5) $\int_{0}^{1} \mathrm{e}^{\mathrm{x}} \mathrm{d} x$ 还是 $\int_{0}^{1}(1+x) \mathrm{d} x$ ?

## 第二节 微积分基本公式

在第一节中有一个应用定积分定义计算积分的例子. 从这个例子我们看到, 被积函数虽然是简单的二次冥函数 $f(x)=x^{2}$, 但直接按定义来计算它的定积 分已经不是很容易的事. 如果被积函数是其他复杂的函数, 其困难就更大了. 因 此, 我们必须寻求计算定积分的新方法.

下面先从实际问题中寻找解决问题的线索. 为此, 我们对变速直线运动中遇 到的位置函数 $s(t)$ 及速度函数 $v(t)$ 之间的联系作进一步的研究.

## 一、变速直线运动中位置函数与速度函数之间的联系

有一物体在一直线上运动. 在这直线上取定原点、正向及长度单位, 使它成 一数轴. 设时刻 $t$ 时物体所在位置为 $s(t)$, 速度为 $v(t)$ (为了讨论方便起见, 可 以设 $v(t) \geqslant 0)$.

从第一节知道: 物体在时间间隔 $\left[T_{1}, T_{2}\right]$ 内经过的路程可以用速度函数 $v(t)$ 在 $\left[T_{1}, T_{2}\right]$ 上的定积分

$$
\int_{T_{1}}^{T_{2}} v(t) \mathrm{d} t
$$

来表达; 另一方面,这段路程又可以通过位置函数 $s(t)$ 在区间 $\left[T_{1}, T_{2}\right]$ 上的增 量

$$
s\left(T_{2}\right)-s\left(T_{1}\right)
$$

来表达. 由此可见,位置函数 $s(t)$ 与速度函数 $v(t)$ 之间有如下关系:

$$
\int_{T_{1}}^{T_{2}} v(t) \mathrm{d} t=s\left(T_{2}\right)-s\left(T_{1}\right) .
$$

因为 $s^{\prime}(t)=v(t)$, 即位置函数 $s(t)$ 是速度函数 $v(t)$ 的原函数, 所以关系 式(1) 表示, 速度函数 $v(t)$ 在区间 $\left[T_{1}, T_{2}\right]$ 上的定积分等于 $v(t)$ 的原函数 $s(t)$ 在区间 $\left[T_{1}, T_{2}\right]$ 上的增量

$$
s\left(T_{2}\right)-s\left(T_{1}\right) .
$$

上述从变速直线运动的路程这个特殊问题中得出来的关系, 在一定条件下 具有普遍性. 事实上, 我们将在第三目中证明, 如果函数 $f(x)$ 在区间 $[a, b]$ 上连 续, 那么, $f(x)$ 在区间 $[a, b]$ 上的定积分就等于 $f(x)$ 的原函数 (设为 $F(x)$ ) 在 区间 $[a, b]$ 上的增量

$$
F(b)-F(a)
$$

## 二、积分上限的函数及其导数

设函数 $f(x)$ 在区间 $[a, b]$ 上连续,并且设 $x$ 为 $[a, b]$ 上的一点. 我们来考 察 $f(x)$ 在部分区间 $[a, x]$ 上的定积分

$$
\int_{a}^{x} f(x) \mathrm{d} x .
$$

首先, 由于 $f(x)$ 在 $[a, x]$ 上仍旧连续, 因此这个定积分存在.这里, $x$ 既表 示定积分的上限, 又表示积分变量. 因为定积分与积分变量的记法无关, 所以, 为 了明确起见, 可以把积分变量改用其他符号, 例如用 $t$ 表示, 则上面的定积分可 以写成

$$
\int_{a}^{r} f(t) \mathrm{d} t
$$

如果上限 $x$ 在区间 $[a, b]$ 上任意变动, 则对于每一个取定的 $x$ 值, 定积分 有一个对应值, 所以它在 $[a, b]$ 上定义了一个函数, 记作 $\Phi(x)$ :

$$
\Phi(x)=\int_{a}^{x} f(t) \mathrm{d} t \quad(a \leqslant x \leqslant b) .
$$

这个函数 $\Phi(x)$ 具有下面定理 1 所指出的重要性质.

定理 1 如果函数 $f(x)$ 在区间 $[a, b]$ 上连续, 则积分上限的函数

$$
\Phi(x)=\int_{a}^{x} f(t) \mathrm{d} t
$$

在 $[a, b]$ 上可导, 并且它的导数

$$
\Phi^{\prime}(x)=\frac{\mathrm{d}}{\mathrm{d} x} \int_{a}^{x} f(t) \mathrm{d} t=f(x) \quad(a \leqslant x \leqslant b) .
$$

证 若 $x \in(a, b)$, 设 $x$ 获得增量 $\Delta x$, 其绝 对值足够地小, 使得 $x+\Delta x \in(a, b)$, 则 $\Phi(x)$ (图 5-6, 图中 $\Delta x>0$ ) 在: $x+\Delta x$ 处的函数值为

$$
\Phi(x+\Delta x)=\int_{a}^{x+\Delta x} f(t) \mathrm{d} t .
$$

由此得函数的增量

$$
\begin{aligned}
\Delta \Phi & =\Phi(x+\Delta x)-\Phi(x) \\
& =\int_{a}^{x+\Delta x} f(t) \mathrm{d} t-\int_{a}^{x} f(t) \mathrm{d} t \\
& =\int_{a}^{x} f(t) \mathrm{d} t+\int_{t}^{x+\Delta x} f(t) \mathrm{d} t-\int_{a}^{\delta} f(t) \mathrm{d} t \\
& =\int_{x}^{x+\Delta x} f(t) \mathrm{d} t .
\end{aligned}
$$

再应用积分中值定理，即有等式

$$
\Delta \Phi=f(\xi) \Delta x,
$$

这里, $\xi$ 在 $x$ 与 $x+\Delta x$ 之间. 把上式两端各除以 $\Delta x$, 得函数增量与自变量增量 的比值

$$
\frac{\Delta \Phi}{\Delta x}=f(\xi) .
$$

由于假设 $f(x)$ 在 $[a, b]$ 上连续, 而 $\Delta x \rightarrow 0$ 时, $\xi \rightarrow x$, 因此 $\lim _{\Delta x \rightarrow 0} f(\xi)=$ $f(x)$. 于是, 令 $\Delta x \rightarrow 0$ 对上式两端取极限时, 左端的极限也应该存在且等于 $f(x)$. 这就是说, 函数 $\Phi(x)$ 的导数存在,并且

$$
\Phi^{\prime}(x)=f(x) \text {. }
$$

若 $x=a$, 取 $\Delta x>0$, 则同理可证 $\Phi^{\prime} ，(a)=f(a)$; 若 $x=b$, 取 $\Delta x<0$, 则同 理可证 $\Phi^{\prime}-(b)=f(b)$.

定理 1 证毕.

这个定理指出了一个重要结论: 连续函数 $f(x)$ 取变上限 $x$ 的定积分然后 求导, 其结果还原为 $f(x)$ 本身. 联想到原函数的定义, 就可以从定理 1 推知 $\Phi(x)$ 是连续函数 $f(x)$ 的一个原函数. 因此, 我们引出如下的原函数的存在 定理.

定理 2 如果函数 $f(x)$ 在区间 $[a, b]$ 上连续, 则函数

$$
\Phi(x)=\int_{a}^{x} f(t) \mathrm{d} t
$$

就是 $f(x)$ 在 $[a, b]$ 上的一个原函数. 这个定理的重要意义是:一方面肯定了连续函数的原函数是存在的,另一方 面初步地揭示了积分学中的定积分与原函数之间的联系. 因此, 我们就有可能通 过原函数来计算定积分.

## 三、牛顿一莱布尼茨公式

现在我们根据定理 2 来证明一个重要定理,它给出了用原函数计算定积分 的公式.

定理 3 如果函数 $F(x)$ 是连续函数 $f(x)$ 在区间 $[a, b]$ 上的一个原函数, 则

$$
\int_{a}^{b} f(x) \mathrm{d} x=F(b)-F(a) .
$$

证已知函数 $F(x)$ 是连续函数 $f(x)$ 的一个原函数, 又根据定理 2 知道, 积分上限的函数

$$
\Phi(x)=\int_{a}^{x} f(t) \mathrm{d} t
$$

也是 $f(x)$ 的一个原函数. 于是这两个原函数之差 $F(x)-\Phi(x)$ 在 $[a, b]$ 上必 定是某一个常数 $C$ (第四章第一节), 即

$$
F(x)-\Phi(x)=C \quad(a \leqslant x \leqslant b) .
$$

在上式中令 $x=a$, 得 $F(a)-\Phi(a)=C$. 又由 $\Phi(x)$ 的定义式(3) 及上节定 积分的补充规定 (1) 可知 $\Phi(a)=0$, 因此, $C=F(a)$. 以 $F(a)$ 代入 (5) 式中的 $C$, 以 $\int_{a}^{x} f(t) \mathrm{d} t$ 代入 $(5)$ 式中的 $\Phi(x)$, 可得

$$
\int_{a}^{x} f(t) \mathrm{d} t=F(x)-F(a) .
$$

在上式中令 $x=b$, 就得到所要证明的公式(4).

由上节定积分的补充规定 (2) 可知, (4) 式对 $a>b$ 的情形同样成立.

为了方便起见, 以居把 $F(b)-F(a)$ 记成 $[F(x)]_{a}^{b}$, 于是 $(4)$ 式又可写成

$$
\int_{a}^{b} f(x) \mathrm{d} x=[F(x)]_{a}^{b} .
$$

公式 (4)叫做牛顿(Newton) - 莱布尼茨(Leibniz)公式(1). 这个公式进一步揭 示了定积分与被积函数的原函数或不定积分之间的联系. 它表明:一个连续函数

(1) 牛顿 (Isaac Newton, 1642-1727) 英国数学家、物理学家,微积分的基基者.牛顿的微积分学说

在区间 $[a, b]$ 上的定积分等于它的任一个原函数在区间 $[a, b]$ 上的增量. 这就 给定积分提供了一个有效而简便的计算方法,大大简化了定积分的计算手续.

通常也把公式(4)叫做微积分基本公式.

下面我们举几个应用公式(4)来计算定积分的简单例子.

例 1 计算第一节中的定积分 $\int_{0}^{1} x^{2} \mathrm{~d} x$.

解 由于 $\frac{x^{3}}{3}$ 是 $x^{2}$ 的一个原函数,所以按牛顿 - 莱布尼茨公式, 有

$$
\int_{0}^{1} x^{2} \mathrm{~d} x=\left[\frac{x^{3}}{3}\right]_{0}^{1}=\frac{1^{3}}{3}-\frac{0^{3}}{3}=\frac{1}{3}-0=\frac{1}{3} .
$$

例 2 计算 $\int_{-1}^{\sqrt{3}} \frac{\mathrm{d} x}{1+x^{2}}$.

解 由于 $\arctan x$ 是 $\frac{1}{1+x^{2}}$ 的一个原函数, 所以

$$
\begin{aligned}
\int_{-1}^{\sqrt{3}} \frac{\mathrm{d} x}{1+x^{2}} & =[\arctan x]_{-1}^{\sqrt{3}}=\arctan \sqrt{3}-\arctan (-1) \\
& =\frac{\pi}{3}-\left(-\frac{\pi}{4}\right)=\frac{7}{12} \pi .
\end{aligned}
$$

例 3 计算 $\int_{-2}^{-1} \frac{\mathrm{d} x}{x}$.

解 当 $x<0$ 时, $\frac{1}{x}$ 的一个原函数是 $\ln |x|$, 现在积分区间是 $[-2,-1]$, 所 以按牛顿 - 莱布尼茨公式, 有

$$
\int_{-2}^{-1} \frac{\mathrm{d} x}{x}=[\ln |x|]_{-2}^{-1}=\ln 1-\ln 2=-\ln 2 .
$$

通过例 3, 我们应该特别注意: 公式 (4) 中的函数 $F(x)$ 必须是 $f(x)$ 在该积 分区间 $[a, b]$ 上的原函数.

例 4 计算正弦曲线 $y=\sin x$ 在 $[0, \pi]$ 上与 $x$ 轴所围成的平面图形 (图 5-7) 的面 积.

解 这图形是曲边梯形的一个特例. 它 的面积

$$
A=\int_{0}^{\pi} \sin x \mathrm{~d} x .
$$

由于 $-\cos x$ 是 $\sin x$ 的一个原函数, 所以

$$
A=\int_{0}^{\pi} \sin x \mathrm{~d} x=[-\cos x]_{10}^{\pi}=-(-1)-(-1)=2 \text {. }
$$

例 5 汽车以每小时 $36 \mathrm{~km}$ 速度行驶, 到某处需要减速停车. 设汽车以等加 速度 $a=-5 \mathrm{~m} / \mathrm{s}^{2}$ 刹车. 问从开始刹车到停车,汽车驶过了多少距离?

解 首先要算出从开始刹车到停车经过的时间. 设开始刹车的时刻为 $t=0$, 此时汽车速度

$$
v_{0}=36 \mathrm{~km} / \mathrm{h}=\frac{36 \times 1000}{3600} \mathrm{~m} / \mathrm{s}=10 \mathrm{~m} / \mathrm{s} .
$$

刹车后汽车减速行驶, 其速度为

$$
v(t)=v_{0}+a t=10-5 t .
$$

当汽车停住时, 速度 $v(t)=0$, 故从

解得

$$
v(t)=10-5 t=0
$$

$$
t=\frac{10}{5}=2(\mathrm{~s}) \text {. }
$$

于是在这段时间内, 汽车所驶过的距离为

$$
s=\int_{0}^{2} v(t) \mathrm{d} t=\int_{0}^{2}(10-5 t) \mathrm{d} t=\left[10 t-5 \times \frac{t^{2}}{2}\right]_{0}^{2}=10(\mathrm{~m}),
$$

即在刹车后, 汽车需驶过 $10 \mathrm{~m}$ 才能停住.

例 6 证明积分中值定理 : 若函数 $f(x)$ 在闭区间 $[a, b]$ 上连续, 则在开区 间 $(a, b)$ 内至少存在一点 $\xi$, 使

$$
\int_{a}^{b} f(x) \mathrm{d} x=f(\xi)(b-a) \quad(a<\xi<b) .
$$

证 因 $f(x)$ 连续, 故它的原函数存在, 设为 $F(x)$, 即设在 $[a, b]$ 上 $F^{\prime}(x)$ $=f(x)$. 根据牛顿 - 莱布尼茨公式, 有

$$
\int_{a}^{b} f(x) \mathrm{d} x=F(b)-F(a) .
$$

显然函数 $F(x)$ 在区间 $[a, b]$ 上满足微分中值定理的条件, 因此按微分中值 定理,在开区间 $(a, b)$ 内至少存在一点 $\xi$, 使

$$
F(b)-F(a)=F^{\prime}(\xi)(b-a), \xi \in(a, b),
$$

故

$$
\int_{a}^{b} f(x) \mathrm{d} x=f(\xi)(b-a), \xi \in(a, b) .
$$

本例的结论是上一节所述积分中值定理的改进. 从本例的证明中不难看出 积分中值定理与微分中值定理的联系.

下面再举几个应用公式 (2) 的例子.

例 7 设 $f(x)$ 在 $[0,+\infty)$ 内连续且 $f(x)>0$. 证明函数

$$
F(x)=\frac{\int_{0}^{x} t f(t) \mathrm{d} t}{\int_{0}^{x} f(t) \mathrm{d} t}
$$

在 $(0,+\infty)$ 内为单调增加函数.

证 由公式(2),得

$$
\frac{\mathrm{d}}{\mathrm{d} x} \int_{0}^{x} t f(t) \mathrm{d} t=x f(x), \quad \frac{\mathrm{d}}{\mathrm{d} x} \int_{1}^{x} f(t) \mathrm{d} t=f(x) .
$$

故

$$
\begin{aligned}
F^{\prime}(x) & =\frac{x f(x) \int_{0}^{x} f(t) \mathrm{d} t-f(x) \int_{0}^{x} t f(t) \mathrm{d} \iota}{\left(\int_{0}^{x} f(t) \mathrm{d} t\right)^{2}} \\
& =\frac{f(x) \int_{0}^{x}(x-\iota) f(t) \mathrm{d} \iota}{\left(\int_{0}^{x} f(t) \mathrm{d} t\right)^{2}} .
\end{aligned}
$$

按假设, 当 $0<t<x$ 时 $f(t)>0,(x-\imath) f(t)>0$, 按例 6 所述积分中值定理可 知

$$
\int_{11}^{t} f(t) \mathrm{d} \iota>0, \quad \int_{11}^{t}(x-\imath) f(\iota) \mathrm{d} \ell>0,
$$

所以 $F^{\prime}(x)>0(x>0)$, 从而 $F(x)$ 在 $(0,+\infty)$ 内为单调增加函数.

例 8 求 $\lim _{x \rightarrow 0} \frac{\int_{\cos x}^{1} \mathrm{e}^{-t^{2}} \mathrm{~d} t}{x^{2}}$.

解 易知这是一个 $\frac{0}{0}$ 型的未定式,我们利用洛必达法则来计算. 分子可写成

$$
-\int_{1}^{\cos x} \mathrm{e}^{-t^{2}} \mathrm{~d} t,
$$

它是以 $\cos x$ 为上限的积分, 作为 $x$ 的函数可看成是以 $u=\cos x$ 为中间变量的 复合函数,故由公式(2)有

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d} x} \int_{\cos x}^{1} \mathrm{e}^{-t^{2}} \mathrm{~d} t & =-\frac{\mathrm{d}}{\mathrm{d} x} \int_{1}^{\cos x} \mathrm{e}^{-t^{2}} \mathrm{~d} t \\
& =-\left.\frac{\mathrm{d}}{\mathrm{d} u} \int_{1}^{\prime \prime} \mathrm{e}^{-t^{2}} \mathrm{~d} t\right|_{u \text { um } x} \cdot(\cos x)^{\prime} \\
& =-\mathrm{e}^{-\cos ^{2} x} \cdot(-\sin x) \\
& =\sin x \mathrm{e}^{-\cos ^{2} t} .
\end{aligned}
$$

因此

$$
\lim _{x \rightarrow 0} \frac{\int_{\mathrm{ax} . \mathrm{r}}^{1} \mathrm{e}^{-t^{2}} \mathrm{~d} t}{x^{2}}=\lim _{x \rightarrow 0} \frac{\sin x \mathrm{e}^{\mathrm{cm} x^{2} x}}{2 x}=\frac{1}{2 \mathrm{e}} .
$$

## 习 题 5-2

1. 试求函数 $y=\int_{0}^{x} \sin t \mathrm{~d} t$ 当 $x=0$ 及 $x=\frac{\pi}{4}$ 时的导数.
2. 求由参数表达式 $x=\int_{0}^{t} \sin u \mathrm{~d} u, y=\int_{0}^{t} \cos u \mathrm{~d} u$ 所确定的函数对 $x$ 的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.
3. 求由 $\int_{0}^{3} \mathrm{e}^{t} \mathrm{~d} t+\int_{0}^{x} \cos t \mathrm{~d} t=0$ 所决定的隐函数对 $x$ 的导数 $\frac{\mathrm{d} y}{\mathrm{~d} x}$.
4. 当 $x$ 为何值时, 函数 $I(x)=\int_{0}^{1} t \mathrm{e}^{-t^{2}} \mathrm{~d} t$ 有极值?
5. 计算下列各导数:
(1) $\frac{\mathrm{d}}{\mathrm{d} x} \int_{0}^{s^{2}} \sqrt{1+t^{2}} \mathrm{~d} t$;
(2) $\frac{\mathrm{d}}{\mathrm{d} x} \int_{0^{2}}^{x^{3}} \frac{\mathrm{d} t}{\sqrt{1+t^{4}}}$;
(3) $\frac{\mathrm{d}}{\mathrm{d} x} \int_{\sin x}^{\cos x} \cos \left(\pi t^{2}\right) \mathrm{d} t$.
6. 计算下列各定积分:
(1) $\int_{0}^{11}\left(3 x^{2}-x+1\right) \mathrm{d} x$;
(2) $\int_{1}^{2}\left(x^{2}+\frac{1}{x^{4}}\right) \mathrm{d} x$;
(3) $\int_{4}^{y} \sqrt{x}(1+\sqrt{x}) \mathrm{d} x$;
(4) $\int_{\frac{1}{\sqrt{3}}}^{\sqrt{3}} \frac{d x}{1+x^{2}}$;
(5) $\int_{-\frac{1}{2}}^{\frac{1}{2}} \frac{\mathrm{d} x}{\sqrt{1-x^{2}}}$;
(6) $\int_{11}^{\sqrt{3} a} \frac{\mathrm{d} x}{a^{2}+x^{2}}$;
(7) $\int_{0}^{1} \frac{\mathrm{d} x}{\sqrt{4-x^{2}}}$;
(8) $\int_{-1}^{0} \frac{3 x^{4}+3 x^{2}+1}{x^{2}+1} \mathrm{~d} x$;
(9) $\int_{-\mathrm{e}-1}^{-2} \frac{\mathrm{d} x}{1+x}$;
(10) $\int_{0}^{\frac{\pi}{4}} \tan ^{2} \theta \mathrm{d} \theta$;
(11) $\int_{11}^{2 x}|\sin x| \mathrm{d} x$;
(12) $\int_{0}^{2} f(x) \mathrm{d} x$, 其中 $f(x)= \begin{cases}x+1, & x \leqslant 1, \\ \frac{1}{2} x^{2}, & x>1 \text {. }\end{cases}$
7. 设 $k \in \mathbf{N}^{*}$. 试证下列各题：
(1) $\int_{-\pi}^{\pi} \cos k x \mathrm{~d} x=0$;
(2) $\int_{-\pi}^{\pi} \sin k x \mathrm{~d} x=0$;
(3) $\int_{-\pi}^{\pi} \cos ^{2} k x \mathrm{~d} x=\pi$;
(4) $\int_{-\pi}^{\pi} \sin ^{2} k x \mathrm{~d} x=\pi$.
8. 设 $k 、 l \in \mathbf{N}^{+}$, 且 $k \neq l$. 证明：
(1) $\int_{-\pi}^{\pi} \cos k x \sin l x \mathrm{~d} x=0$;
(2) $\int_{-\pi}^{\pi} \cos k x \cos l x \mathrm{~d} x=0$;

(3) $\int_{-\pi}^{\pi} \sin k x \sin l x \mathrm{~d} x=0$.

9. 求下列极限:
(1) $\lim _{x \rightarrow 11} \frac{\int_{11}^{x} \cos t^{2} \mathrm{~d} t}{x}$;
(2) $\lim _{x \rightarrow 0} \frac{\left(\int_{0}^{x} \mathrm{e}^{t^{2}} \mathrm{~d} t\right)^{2}}{\int_{0}^{x} t \mathrm{e}^{2 t^{2}} \mathrm{~d} t}$.
10. 设

$$
f(x)= \begin{cases}x^{2}, & x \in[0,1), \\ x, & x \in[1,2] .\end{cases}
$$

求 $\Phi(x)=\int_{0}^{x} f(t) \mathrm{d} t$ 在 $[0,2]$ 上的表达式,并讨论 $\Phi(x)$ 在 $(0,2)$ 内的连续性.

11. 设

$$
f(x)=\left\{\begin{array}{cc}
\frac{1}{2} \sin x, & 0 \leqslant x \leqslant \pi, \\
0, & x<0 \text { 或 } x>\pi .
\end{array}\right.
$$

求 $\Phi(x)=\int_{0}^{1} f(t) \mathrm{d} t$ 在 $(-\infty,+\infty)$ 内的表达式.

12. 设 $f(x)$ 在 $[a, b]$ 上连续,在 $(a, b)$ 内可异且 $f^{\prime}(x) \leqslant 0$,

$$
F(x)=\frac{1}{x-a} \int_{a}^{x} f(t) \mathrm{d} t .
$$

证明在 $(a, b)$ 内有 $F^{\prime}(x) \leqslant 0$.

13. 设 $F(x)=\int_{0}^{t} \frac{\sin t}{t} \mathrm{~d} t$, 求 $F^{\prime}(\dot{0})$.
14. 设 $f(x)$ 在 $[0,+\infty)$ 内连续, 且 $\lim _{, \rightarrow+\infty} f(x)=1$. 证明函数

$$
y=\mathrm{e}^{-x} \int_{11}^{x} \mathrm{e}^{\prime} f(t) \mathrm{d} t
$$

满足方程 $\frac{\mathrm{d} y}{\mathrm{~d} x}+y=f(x)$. 并求 $\lim _{x \rightarrow+\infty} y(x)$.

## 第三节 定积分的换元法和分部积分法

由上节结果知道, 计算定积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 的简便方法是把它转化为求 $f(x)$ 的原函数的增量. 在第四章中, 我们知道用换元积分法和分部积分法可以求出一 些函数的原函数. 因此, 在一定条件下, 可以用换元积分法和分部积分法来计算 定积分.下面就来讨论定积分的这两种计算方法.

## 一、定积分的换元法

为了说明如何用换元法来计算定积分,先证明下面的定理.

定理 假设函数 $f(x)$ 在区间 $[a, b]$ 上连续,函数 $x=\varphi(t)$ 满足条件:

(1) $\varphi(a)=a, \varphi(\beta)=b$; (2) $\varphi(t)$ 在 $[\alpha, \beta]$ (或 $[\beta, \alpha]$ ) 上具有连续导数,且其值域 $R_{\varphi}=[a, b] \mathbb{1}$, 则有

$$
\int_{a}^{\prime \prime} f(x) \mathrm{d} x=\int_{a}^{\beta} f[\varphi(t)] \varphi^{\prime}(t) \mathrm{d} t .
$$

公式(1)叫做定积分的换元公式.

证 由假设可以知道, 上式两边的被积函数都是连续的, 因此不仅上式两边 的定积分都存在,而且由上节的定理 2 知道, 被积函数的原函数也都存在.所以, (1) 式两边的定积分都可应用牛顿 - 莱布尼茨公式. 假设 $F(x)$ 是 $f(x)$ 的一个 原函数, 则

$$
\int_{a}^{b} f(x) \mathrm{d} x=F(b)-F(a) .
$$

另一方面, 记 $\Phi(t)=F[\varphi(t)]$, 它是由 $F(x)$ 与 $x=\varphi(t)$ 复合而成的函数. 由复 合函数求导法则, 得

$$
\Phi^{\prime}(t)=\frac{\mathrm{d} F}{\mathrm{~d} x} \mathrm{~d} x=f(x) \varphi^{\prime}(t)=f[\varphi(t)] \varphi^{\prime}(t) .
$$

这表明 $\Phi(t)$ 是 $f[\varphi(t)] \varphi^{\prime}(t)$ 的一个原函数. 因此有

$$
\int_{a}^{\beta} f[\varphi(t)] \varphi^{\prime}(t) \mathrm{d} t=\Phi(\beta)-\Phi(\alpha) .
$$

又由 $\Phi(t)=F[\varphi(t)]$ 及 $\varphi(\alpha)=a, \varphi(\beta)=b$ 可知

$$
\Phi(\beta)-\Phi(\alpha)=F[\varphi(\beta)]-F[\varphi(\alpha)]=F(b)-F(a) \text {. }
$$

所以

$$
\begin{aligned}
\int_{a}^{b} f(x) \mathrm{d} x & =F(b)-F(a)=\Phi(\beta)-\Phi(\alpha) \\
& =\int_{a}^{\beta} f[\varphi(t)] \varphi^{\prime}(t) \mathrm{d} t .
\end{aligned}
$$

这就证明了换元公式.

在定积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 中的 $\mathrm{d} x$, 本来是整个定积分记号中不可分割的一部 分, 但由上述定理可知, 在一定条件下, 它确实可以作为微分记号来对待. 这就是 说,应用换元公式时,如果把 $\int_{a}^{t} f(x) \mathrm{d} x$ 中的 $x$ 换成 $\varphi(t)$, 则 $\mathrm{d} x$ 就换成 $\varphi^{\prime}(t) \mathrm{d} t$, 这正好是 $x=\varphi(t)$ 的微分 $\mathrm{d} x$.

应用换元公式时有两点值得注意: (1) 用 $x=\varphi(t)$ 把原来变量 $x$ 代换成新 变量 $t$ 时,积分限也要换成相应于新变量 $t$ 的积分限; (2) 求出 $f[\varphi(t)] \varphi^{\prime}(t)$

论仍成立. 的一个原函数 $\Phi(t)$ 后, 不必像计算不定积分那样再要把 $\Phi(t)$ 变换成原来变量 $x$ 的函数,而只要把新变量 $t$ 的上、下限分别代入 $\Phi(t)$ 中然后相减就行了.

例 1 计算 $\int_{0}^{a} \sqrt{a^{2}-x^{2}} \mathrm{~d} x \quad(a>0)$.

解 设 $x=a \sin t$, 则 $\mathrm{d} x=a \cos t \mathrm{~d} t$,

$$
\text { 当 } x=0 \text { 时, 取 } t=0 \text {; 当 } x=a \text { 时, 取 } t=\frac{\pi}{2} \text {. }
$$

于是

$$
\begin{aligned}
\int_{0}^{a} \sqrt{a^{2}-x^{2}} \mathrm{~d} x & =a^{2} \int_{0}^{\frac{\pi}{2}} \cos ^{2} t \mathrm{~d} t=\frac{a^{2}}{2} \int_{0}^{\frac{\pi}{2}}(1+\cos 2 t) \mathrm{d} t \\
& =\frac{a^{2}}{2}\left[t+\frac{1}{2} \sin 2 t\right]_{0}^{\frac{\pi}{2}}=\frac{\pi a^{2}}{4} .
\end{aligned}
$$

换元公式也可反过来使用. 为使用方便起见,把换元公式中左右两边对调位 置, 同时把 $t$ 改记为 $x$, 而 $x$ 改记为 $t$, 得

$$
\int_{a}^{b} f[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x=\int_{a}^{\beta} f(t) \mathrm{d} t .
$$

这样, 我们可用 $t=\varphi(x)$ 来引人新变量 $t$, 而 $\alpha=\varphi(a), \beta=\varphi(b)$.

例 2 计算 $\int_{0}^{\frac{\pi}{2}} \cos ^{5} x \sin x \mathrm{~d} x$.

解 设 $t=\cos x$, 则 $\mathrm{d} t=-\sin x \mathrm{~d} x$, 且

$$
\text { 当 } x=0 \text { 时, } t=1 \text {; 当 } x=\frac{\pi}{2} \text { 时, } t=0 \text {. }
$$

于是

$$
\int_{0}^{\frac{\pi}{2}} \cos ^{5} x \sin x \mathrm{~d} x=-\int_{1}^{0} t^{5} \mathrm{~d} t=\int_{0}^{1} t^{5} \mathrm{~d} t=\left[\frac{t^{6}}{6}\right]_{0}^{1}=\frac{1}{6} .
$$

在例 2 中, 如果我们不明显地写出新变量 $t$, 那么定积分的上、下限就不要 变更. 现在用这种记法写出计算过程如下:

$$
\begin{aligned}
\int_{0}^{\frac{\pi}{2}} \cos ^{5} x \sin x \mathrm{~d} x & =-\int_{0}^{\frac{\pi}{2}} \cos ^{5} x \mathrm{~d}(\cos x) \\
& =-\left[\frac{\cos ^{6} x}{6}\right]_{0}^{\frac{\pi}{2}}=-\left(0-\frac{1}{6}\right)=\frac{1}{6} .
\end{aligned}
$$

例 3 计算 $\int_{0}^{\pi} \sqrt{\sin ^{3} x-\sin ^{5} x} \mathrm{~d} x$.

解 由于 $\sqrt{\sin ^{3} x-\sin ^{5} x}=\sqrt{\sin ^{3} x\left(1-\sin ^{2} x\right)}=\sin ^{\frac{3}{2}} x \cdot|\cos x|$, 在 $\left[0, \frac{\pi}{2}\right]$ 上, $|\cos x|=\cos x$; 在 $\left[\frac{\pi}{2}, \pi\right]$ 上, $|\cos x|=-\cos x$, 所以

$$
\begin{aligned}
\int_{0}^{\pi} \sqrt{\sin ^{3} x-\sin ^{5} x} \mathrm{~d} x & =\int_{0}^{\frac{\pi}{2}} \sin ^{\frac{3}{2}} x \cos x \mathrm{~d} x+\int_{\frac{\pi}{2}}^{\pi} \sin ^{\frac{3}{2}} x(-\cos x) \mathrm{d} x \\
& =\int_{0}^{\frac{\pi}{2}} \sin ^{\frac{3}{2}} x \mathrm{~d}(\sin x)-\int_{\frac{\pi}{2}}^{\pi} \sin ^{\frac{3}{2}} x \mathrm{~d}(\sin x) \\
& =\left[\frac{2}{5} \sin ^{\frac{5}{2}} x\right]_{0}^{\frac{\pi}{2}}-\left[\frac{2}{5} \sin ^{\frac{5}{2}} x\right]_{\frac{\pi}{2}}^{\pi} \\
& =\frac{2}{5}-\left(-\frac{2}{5}\right)=\frac{4}{5} .
\end{aligned}
$$

注意 如果忽略 $\cos x$ 在 $\left[\frac{\pi}{2}, \pi\right]$ 上非正, 而按

计算, 将导致错误.

$$
\sqrt{\sin ^{3} x-\sin ^{5} x}=\sin ^{\frac{3}{2}} x \cos x
$$

例 4 计算 $\int_{0}^{4} \frac{x+2}{\sqrt{2 x+1}} \mathrm{~d} x$.

解 设 $\sqrt{2 x+1}=t$, 则 $x=\frac{t^{2}-1}{2}, \mathrm{~d} x=t \mathrm{~d} t$, 且

$$
\text { 当 } x=0 \text { 时, } t=1 \text {; 当 } x=4 \text { 时, } t=3 \text {. }
$$

于是

$$
\begin{aligned}
\int_{0}^{4} \frac{x+2}{\sqrt{2 x+1}} \mathrm{~d} x & =\int_{1}^{3} \frac{\frac{t^{2}-1}{2}+2}{t} t \mathrm{~d} t=\frac{1}{2} \int_{1}^{3}\left(t^{2}+3\right) \mathrm{d} t \\
& =\frac{1}{2}\left[\frac{t^{3}}{3}+3 t\right]_{1}^{3} \\
& =\frac{1}{2}\left[\left(\frac{27}{3}+9\right)-\left(\frac{1}{3}+3\right)\right]=\frac{22}{3}
\end{aligned}
$$

例 5 证明:

(1) 若 $f(x)$ 在 $[-a, a]$ 上连续且为偶函数, 则

$$
\int_{-a}^{a} f(x) \mathrm{d} x=2 \int_{0}^{a} f(x) \mathrm{d} x .
$$

(2) 若 $f(x)$ 在 $[-a, a]$ 上连续且为奇函数, 则

$$
\int_{-u}^{a} f(x) \mathrm{d} x=0 \text {. }
$$

证因为

$$
\int_{-a}^{a} f(x) \mathrm{d} x=\int_{-a}^{a} f(x) \mathrm{d} x+\int_{11}^{a} f(x) \mathrm{d} x,
$$

对积分 $\int_{-a}^{0} f(x) \mathrm{d} x$ 作代换 $x=-t$, 则得

$$
\int_{-a}^{0} f(x) \mathrm{d} x=-\int_{a}^{0} f(-t) \mathrm{d} t=\int_{0}^{a} f(-t) \mathrm{d} t=\int_{0}^{a} f(-x) \mathrm{d} x .
$$

于是

$$
\begin{aligned}
\int_{-a}^{a} f(x) \mathrm{d} x & =\int_{0}^{a} f(-x) \mathrm{d} x+\int_{0}^{a} f(x) \mathrm{d} x \\
& =\int_{0}^{a}[f(x)+f(-x)] \mathrm{d} x .
\end{aligned}
$$

(1) 若 $f(x)$ 为偶函数, 则

从而

$$
f(x)+f(-x)=2 f(x),
$$

$$
\int_{-a}^{a} f(x) \mathrm{d} x=2 \int_{0}^{a} f(x) \mathrm{d} x .
$$

(2) 若 $f(x)$ 为奇函数, 则

$$
f(x)+f(-x)=0 \text {, }
$$

从而

$$
\int_{-a}^{a} f(x) \mathrm{d} x=0 \text {. }
$$

利用例 5 的结论,常可简化计算偶函数、奇函数在对称于原点的区间上的定 积分.

例 6 若 $f(x)$ 在 $[0,1]$ 上连续, 证明

(1) $\int_{0}^{\frac{\pi}{2}} f(\sin x) \mathrm{d} x=\int_{0}^{\frac{\pi}{2}} f(\cos x) \mathrm{d} x$;

(2) $\int_{0}^{\pi} x f(\sin x) \mathrm{d} x=\frac{\pi}{2} \int_{0}^{\pi} f(\sin x) \mathrm{d} x$, 由此计算

$$
\int_{0}^{\pi} \frac{x \sin x}{1+\cos ^{2} x} \mathrm{~d} x \text {. }
$$

证（1）设 $x=\frac{\pi}{2}-t$, 则 $\mathrm{d} x=-\mathrm{d} t$, 且

$$
\text { 当 } x=0 \text { 时, } t=\frac{\pi}{2} \text {; 当 } x=\frac{\pi}{2} \text { 时, } t=0 \text {. }
$$

于是

$$
\begin{aligned}
\int_{0}^{\frac{\pi}{2}} f(\sin x) \mathrm{d} x & =-\int_{\frac{\pi}{2}}^{0} f\left[\sin \left(\frac{\pi}{2}-t\right)\right] \mathrm{d} t \\
& =\int_{0}^{\frac{\pi}{2}} f(\cos t) \mathrm{d} t=\int_{0}^{\frac{\pi}{2}} f(\cos x) \mathrm{d} x .
\end{aligned}
$$

（2）设 $x=\pi-t$, 则 $\mathrm{d} x=-\mathrm{d} t$, 且

当 $x=0$ 时, $t=\pi$; 当 $x=\pi$ 时, $t=0$.

于是

$$
\begin{aligned}
\int_{0}^{\pi} x f(\sin x) \mathrm{d} x & =-\int_{\pi}^{0}(\pi-t) f[\sin (\pi-t)] \mathrm{d} t \\
& =\int_{0}^{\pi}(\pi-t) f(\sin t) \mathrm{d} t \\
& =\pi \int_{0}^{\pi} f(\sin t) \mathrm{d} t-\int_{0}^{\pi} t f(\sin t) \mathrm{d} t \\
& =\pi \int_{0}^{\pi} f(\sin x) \mathrm{d} x-\int_{0}^{\pi} x f(\sin x) \mathrm{d} x,
\end{aligned}
$$

所以

$$
\int_{0}^{\pi} x f(\sin x) \mathrm{d} x=\frac{\pi}{2} \int_{0}^{\pi} f(\sin x) \mathrm{d} x .
$$

利用上述结论, 即得

$$
\begin{aligned}
\int_{0}^{\pi} \frac{x \sin x}{1+\cos ^{2} x} \mathrm{~d} x & =\frac{\pi}{2} \int_{0}^{\pi} \frac{\sin x}{1+\cos ^{2} x} \mathrm{~d} x=-\frac{\pi}{2} \int_{0}^{\pi} \frac{\mathrm{d}(\cos x)}{1+\cos ^{2} x} \\
& =-\frac{\pi}{2}[\arctan (\cos x)]_{0}^{\pi} \\
& =-\frac{\pi}{2}\left(-\frac{\pi}{4}-\frac{\pi}{4}\right)=\frac{\pi^{2}}{4} .
\end{aligned}
$$

例 7 设 $f(x)$ 是连续的周期函数, 周期为 $T$, 证明:

(1) $\int_{a}^{a+T} f(x) \mathrm{d} x=\int_{0}^{T} f(x) \mathrm{d} x$;

(2) $\int_{a}^{a+n \mathrm{~T}} f(x) \mathrm{d} x=n \int_{0}^{T} f(x) \mathrm{d} x \quad(n \in \mathrm{N})$, 由此计算

$$
\int_{0}^{n \pi} \sqrt{1+\sin 2 x} \mathrm{~d} x
$$

证 (1) 记 $\Phi(a)=\int_{a}^{a+T} f(x) \mathrm{d} x$, 则 $\Phi^{\prime}(a)=f(a+T)-f(a)=0$, 知 $\Phi(a)$ 与 $a$ 无关, 因此 $\Phi(a)=\Phi(0)$, 即

$$
\int_{a}^{a+T} f(x) \mathrm{d} x=\int_{0}^{T} f(x) \mathrm{d} x .
$$

(2) $\int_{a}^{u+n T} f(x) \mathrm{d} x=\sum_{k=0}^{n-1} \int_{a+k T}^{a+k T+T} f(x) \mathrm{d} x$,

由(1) 知 $\int_{a+k T}^{a+k T+T} f(x) \mathrm{d} x=\int_{0}^{T} f(x) \mathrm{d} x$, 因此

$$
\int_{a}^{u+n T} f(x) \mathrm{d} x=n \int_{0}^{T} f(x) \mathrm{d} x .
$$

由于 $\sqrt{1+\sin 2 x}$ 是以 $\pi$ 为周期的周期函数, 利用上述结论, 有

$$
\int_{0}^{n \pi} \sqrt{1+\sin 2 x} \mathrm{~d} x=n \int_{0}^{\pi} \sqrt{1+\sin 2 x} \mathrm{~d} x
$$

$$
\begin{aligned}
& =n \int_{0}^{\pi}|\sin x+\cos x| \mathrm{d} x \\
& =\sqrt{2} n \int_{0}^{r}\left|\sin \left(x+\frac{\pi}{4}\right)\right| \mathrm{d} x \\
& =\sqrt{2} n \int_{\frac{\pi}{4}}^{\frac{5 \pi}{4}}|\sin t| \mathrm{d} t \\
& =\sqrt{2} n \int_{0}^{r}|\sin t| \mathrm{d} t \\
& =\sqrt{2} n \int_{0}^{\pi} \sin t d t \\
& =2 \sqrt{2} n .
\end{aligned}
$$

例 8 计算 $\int_{0}^{3} \frac{x^{2}}{\left(x^{2}-3 x+3\right)^{2}} \mathrm{~d} x$.

解 $x^{2}-3 x+3=\left(x-\frac{3}{2}\right)^{2}+\frac{3}{4}$, 令 $x-\frac{3}{2}=\frac{\sqrt{3}}{2} \tan u\left(|u|<\frac{\pi}{2}\right)$, 则 $\left(x^{2}-3 x+3\right)^{2}=\left(\frac{3}{4} \sec ^{2} u\right)^{2}=\frac{9}{16} \sec ^{4} u, \mathrm{~d} x=\frac{\sqrt{3}}{2} \sec ^{2} u \mathrm{~d} u$. 且

$$
\text { 当 } x=0 \text { 对, } u=-\frac{\pi}{3} ; x=3 \text { 付, } u=\frac{\pi}{3} \text {. }
$$

于是 $\int_{0}^{3} \frac{x^{2}}{\left(x^{2}-3 x+3\right)^{2}} \mathrm{~d} x=\int_{-\frac{\pi}{3}}^{\frac{\pi}{3}}\left(\frac{3}{4} \tan ^{2} u+\frac{3 \sqrt{3}}{2} \tan u+\frac{9}{4}\right) \cdot \frac{16}{9} \cdot \frac{\sqrt{3}}{2} \cos ^{2} u \mathrm{~d} u$

$$
\begin{aligned}
& =\frac{8}{3 \sqrt{3}} \cdot 2 \int_{0}^{\frac{\pi}{3}}\left(\frac{3}{4} \tan ^{2} u+\frac{9}{4}\right) \cos ^{2} u \mathrm{~d} u \\
& =\frac{4}{\sqrt{3}} \int_{0}^{\frac{\pi}{3}}\left(\sin ^{2} u+3 \cos ^{2} u\right) \mathrm{d} u \\
& =\frac{4}{\sqrt{3}} \int_{0}^{\frac{\pi}{3}}(2+\cos 2 u) \mathrm{d} u \\
& =\frac{4}{\sqrt{3}}\left[2 u+\frac{1}{2} \sin 2 u\right]_{0}^{\frac{\pi}{3}} \\
& =\frac{8 \pi}{3 \sqrt{3}}+1
\end{aligned}
$$

例 9 设函数

$$
f(x)=\left\{\begin{array}{cc}
x \mathrm{e}^{-x^{2}}, & x \geqslant 0, \\
\frac{1}{1+\cos x^{2}}, & -\pi<x<0,
\end{array}\right.
$$

计算 $\int_{1}^{4} f(x-2) \mathrm{d} x$ ：

解 设 $x-2=t$, 则 $\mathrm{d} x=\mathrm{d} t$, 且

$$
\text { 当 } x=1 \text { 时, } t=-1 \text {; 当 } x=4 \text { 时, } t=2 \text {. }
$$

于是

$$
\begin{aligned}
\int_{1}^{4} f(x-2) \mathrm{d} x & =\int_{-1}^{2} f(t) \mathrm{d} t=\int_{-1}^{0} \frac{\mathrm{d} t}{1+\cos t}+\int_{0}^{2} t \mathrm{e}^{-t^{2}} \mathrm{~d} t \\
& =\left[\tan \frac{t}{2}\right]_{-1}^{0}-\left[\frac{1}{2} \mathrm{e}^{-t^{2}}\right]_{0}^{2} \\
& =\tan \frac{1}{2}-\frac{1}{2} \mathrm{e}^{-4}+\frac{1}{2}
\end{aligned}
$$

## 二、定积分的分部积分法

依据不定积分的分部积分法, 可得

简记作

$$
\begin{aligned}
\int_{u}^{b} u(x) v^{\prime}(x) \mathrm{d} x & =\left[\int u(x) v^{\prime}(x) \mathrm{d} x\right]_{a}^{b} \\
& =\left[u(x) v(x)-\int v(x) u^{\prime}(x) \mathrm{d} x\right]_{a}^{\prime} \\
& =[u(x) v(x)]_{a}^{b}-\int_{a}^{b} v(x) u^{\prime}(x) \mathrm{d} x,
\end{aligned}
$$

或

$$
\int_{a}^{b} u v^{\prime} \mathrm{d} x=[u v]_{a}^{b}-\int_{a}^{b} v u^{\prime} \mathrm{d} x,
$$

$$
\int_{a}^{u} u \mathrm{~d} v=[u v]_{u}^{b}-\int_{a}^{b} v \mathrm{~d} u .
$$

这就是定积分的分部积分公式. 公式表明原函数已经积出的部分可以先用 上、下限代入.

例 10 计算 $\int_{0}^{\frac{1}{2}} \arcsin x \mathrm{~d} x$.

解 $\int_{0}^{\frac{1}{2}} \arcsin x \mathrm{~d} x=[x \arcsin x]_{0}^{\frac{1}{2}}-\int_{0}^{\frac{1}{2}} \frac{x}{\sqrt{1-x^{2}}} \mathrm{~d} x$

$$
=\frac{1}{2} \cdot \frac{\pi}{6}+\left[\sqrt{1-x^{2}}\right]_{0}^{\frac{1}{2}}=\frac{\pi}{12}+\frac{\sqrt{3}}{2}-1 .
$$

例 11 计算 $\int_{0}^{1} \mathrm{e}^{\sqrt{x}} \mathrm{~d} x$.

解 先用换元法. 令 $\sqrt{x}=t$, 则 $x=t^{2}, \mathrm{~d} x=2 t \mathrm{~d} t$, 且

当 $x=0$ 时, $\iota=0$; 当 $x=1$ 时, $\iota=1$. 于是 $\int_{0}^{1} \mathrm{e}^{\sqrt{x}} \mathrm{~d} x=2 \int_{0}^{1} t \mathrm{e}^{\prime} \mathrm{d} t=2 \int_{0}^{1} t \mathrm{~d}\left(\mathrm{e}^{t}\right)$

$$
\begin{aligned}
& =2\left(\left[t \mathrm{e}^{t}\right]_{0}^{1}-\int_{0}^{1} \mathrm{e}^{t} \mathrm{~d} t\right)=2\left(\mathrm{e}-\left[\mathrm{e}^{t}\right]_{0}^{1}\right) \\
& =2[\mathrm{e}-(\mathrm{e}-1)]=2 .
\end{aligned}
$$

例 12 证明定积分公式 (见附录 III 积分表公式(147))：

$$
\begin{aligned}
I_{n} & =\int_{0}^{\frac{\pi}{2}} \sin ^{\prime \prime} x \mathrm{~d} x\left(=\int_{0}^{\frac{\pi}{2}} \cos ^{\prime \prime} x \mathrm{~d} x\right) \\
& =\left\{\begin{array}{l}
\frac{n-1}{n} \cdot \frac{n-3}{n-2} \cdots \cdot \frac{3}{4} \cdot \frac{1}{2} \cdot \frac{\pi}{2}, n \text { 为正偶数, } \\
\frac{n-1}{n} \cdot \frac{n-3}{n-2} \cdots \cdots \cdot \frac{4}{5} \cdot \frac{2}{3}, n \text { 为大于 } 1 \text { 的正奇数. }
\end{array}\right.
\end{aligned}
$$

证

$$
\begin{aligned}
I_{n} & =-\int_{0}^{\frac{\pi}{2}} \sin ^{n-1} x \mathrm{~d}(\cos x) \\
& =\left[-\cos x \sin ^{n-1} x\right]_{0}^{\frac{\pi}{2}}+(n-1) \int_{0}^{\frac{\pi}{2}} \sin ^{n-2} x \cos ^{2} x \mathrm{~d} x .
\end{aligned}
$$

右端第一项等于零; 将第二项里的 $\cos ^{2} x$ 写成 $1-\sin ^{2} x$, 并把积分分成两个, 得

$$
\begin{aligned}
I_{n} & =(n-1) \int_{0}^{\frac{\pi}{2}} \sin ^{n-2} x \mathrm{~d} x-(n-1) \int_{0}^{\frac{\pi}{2}} \sin ^{n} x \mathrm{~d} x \\
& =(n-1) I_{n-2}-(n-1) I_{n},
\end{aligned}
$$

由此得

$$
I_{n}=\frac{n-1}{n} I_{n-2} \text {. }
$$

这个等式叫做积分 $I_{n}$ 关于下标的递推公式.

如果把 $n$ 换成 $n-2$, 则得

$$
I_{n-2}=\frac{n-3}{n-2} I_{n-4} \text {. }
$$

同样地依次进行下去, 直到 $I_{n}$ 的下标递减到 0 或 1 为止.于是,

$$
\begin{aligned}
& I_{2 m}=\frac{2 m-1}{2 m} \cdot \frac{2 m-3}{2 m-2} \cdots \cdot \frac{5}{6} \cdot \frac{3}{4} \cdot \frac{1}{2} I_{0}, \\
& I_{2 m+1}=\frac{2 m}{2 m+1} \cdot \frac{2 m-2}{2 m-1} \cdots \cdot \frac{6}{7} \cdot \frac{4}{5} \cdot \frac{2}{3} I_{1} \quad(m=1,2, \cdots),
\end{aligned}
$$

而

$$
I_{0}=\int_{0}^{\frac{\pi}{2}} \mathrm{~d} x=\frac{\pi}{2}, \quad I_{1}=\int_{0}^{\frac{\pi}{2}} \sin x \mathrm{~d} x=1,
$$

因此

$$
\begin{gathered}
I_{2 m}=\frac{2 m-1}{2 m} \cdot \frac{2 m-3}{2 m-2} \cdots \cdot \frac{5}{6} \cdot \frac{3}{4} \cdot \frac{1}{2} \cdot \frac{\pi}{2}, \\
I_{2 m+1}=\frac{2 m}{2 m+1} \cdot \frac{2 m-2}{2 m-1} \cdots \cdots \frac{6}{7} \cdot \frac{4}{5} \cdot \frac{2}{3}(m=1,2, \cdots) .
\end{gathered}
$$

至于定积分 $\int_{0}^{\frac{\pi}{2}} \cos ^{n} x \mathrm{~d} x$ 与 $\int_{0}^{\frac{\pi}{2}} \sin ^{n} x \mathrm{~d} x$ 相等, 由本节例 6(1) 即可知道, 证毕.

## 习 题 5-3

## 1. 计算下列定积分：

(1) $\int_{\frac{\pi}{3}}^{\pi} \sin \left(x+\frac{\pi}{3}\right) \mathrm{d} x$;
(2) $\int_{-2}^{1} \frac{\mathrm{d} x}{(11+5 x)^{3}}$;
(3) $\int_{0}^{\frac{\pi}{2}} \sin \varphi \cos ^{3} \varphi \mathrm{d} \varphi$;
(4) $\int_{0}^{\pi}\left(1-\sin ^{3} \theta\right) \mathrm{d} \theta$;
(5) $\int_{\frac{\pi}{6}}^{\frac{\pi}{2}} \cos ^{2} u \mathrm{~d} u$;
(6) $\int_{0}^{\sqrt{2}} \sqrt{2-x^{2}} \mathrm{~d} x$;
(7) $\int_{\sqrt{2}}^{\sqrt{2}} \sqrt{8-2 y^{2}} \mathrm{~d} y$;
(8) $\int_{\frac{1}{\sqrt{2}}}^{1} \frac{\sqrt{1-x^{2}}}{x^{2}} \mathrm{~d} x$;
(9) $\int_{0}^{a} x^{2} \sqrt{a^{2}-x^{2}} \mathrm{~d} x(a>0)$;
(10) $\int_{1}^{\sqrt{3}} \frac{\mathrm{d} x}{x^{2} \sqrt{1+x^{2}}}$;
(11) $\int_{-1}^{1} \frac{x \mathrm{~d} x}{\sqrt{5-4 x}}$;
(12) $\int_{1}^{4} \frac{\mathrm{d} x}{1+\sqrt{x}}$;
(13) $\int_{\frac{3}{4}}^{1} \frac{\mathrm{d} x}{\sqrt{1-x}-1}$;
(14) $\int_{u}^{\sqrt{2} a} \frac{x \mathrm{~d} x}{\sqrt{3 a^{2}-x^{2}}}(a>0)$;
(15) $\int_{0}^{1} t \mathrm{e}^{-\frac{t^{2}}{2}} \mathrm{~d} t$;
(16) $\int_{1}^{\mathrm{e}^{2}} \frac{\mathrm{d} x}{x \sqrt{1+\ln x}}$;
(17) $\int_{-2}^{0} \frac{(x+2) \mathrm{d} x}{x^{2}+2 x+2}$;
(18) $\int_{0}^{2} \frac{x d x}{\left(x^{2}-2 x+2\right)^{2}}$;
(19) $\int_{-\pi}^{\pi} x^{4} \sin x \mathrm{~d} x$;
(20) $\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} 4 \cos ^{4} \theta d \theta$;
(21) $\int_{-\frac{1}{2}}^{\frac{1}{2}} \frac{(\arcsin x)^{2}}{\sqrt{1-x^{2}}} \mathrm{~d} x$;
(22) $\int_{-5}^{5} \frac{x^{3} \sin ^{2} x}{x^{4}+2 x^{2}+1} \mathrm{~d} x$;
(23) $\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \cos x \cos 2 x \mathrm{~d} x$;
(24) $\int_{-\frac{\pi}{2}}^{\frac{\pi}{2}} \sqrt{\cos x-\cos ^{3} x} \mathrm{~d} x$;
(25) $\int_{11}^{\pi} \sqrt{1+\cos 2 x} \mathrm{~d} x$;
(26) $\int_{0}^{2 \pi}|\sin (x+1)| \mathrm{d} x$.

2. 设 $f(x)$ 在 $[a, b]$ 上连续, 证明

$$
\int_{a}^{b} f(x) \mathrm{d} x=\int_{a}^{b} f(a+b-x) \mathrm{d} x .
$$

3. 证明: $\int_{x}^{1} \frac{\mathrm{d} x}{1+x^{2}}=\int_{1}^{\frac{1}{x}} \frac{\mathrm{d} x}{1+x^{2}} \quad(x>0)$.
4. 证明: $\int_{11}^{1} x^{m}(1-x)^{n} \mathrm{~d} x=\int_{0}^{1} x^{n}(1-x)^{m} \mathrm{~d} x \quad(m, n \in \mathrm{N})$.
5. 设 $f(x)$ 在 $[0,1]$ 上连续, $n \in Z$, 证明

$$
\int_{\frac{n}{2} \pi}^{\frac{n+1}{2} \pi} f(|\sin x|) \mathrm{d} x=\int_{\frac{n}{2} \pi}^{\frac{n+11}{2} \pi} f(|\cos x|) \mathrm{d} x=\int_{0}^{\frac{\pi}{2}} f(\sin x) \mathrm{d} x .
$$

6. 若 $f(t)$ 是连续的奇函数,证明 $\int_{0}^{x} f(t) \mathrm{d} t$ 是偶函数; 若 $f(t)$ 是连续的偶函数，证明 $\int_{0}^{x} f(t) \mathrm{d} t$ 是奇函数.
7. 计算下列定积分
(1) $\int_{0}^{1} x \mathrm{e}^{-\cdot r} \mathrm{~d} x$;
(2) $\int_{1}^{\mathrm{c}} x \ln x \mathrm{~d} x$;
(3) $\int_{0}^{\frac{2 \pi}{\omega}} t \sin \omega t \mathrm{~d} t \quad(\omega$ 为常数);
(4) $\int_{\frac{\pi}{5}}^{\frac{\pi}{3}} \frac{x}{\sin ^{2} x} \mathrm{~d} x$;
(5) $\int_{1}^{4} \frac{\ln x}{\sqrt{x}} \mathrm{~d} x$;
(6) $\int_{1}^{1} x \arctan x d x$;
(7) $\int_{0}^{\frac{\pi}{2}} \mathrm{e}^{2 x} \cos x \mathrm{~d} x$;
(8) $\int_{1}^{2} x \log _{2} x \mathrm{~d} x$;
(9) $\int_{0}^{\pi}(x \sin x)^{2} \mathrm{~d} x$;
(10) $\int_{1}^{\mathrm{c}} \sin (\ln x) \mathrm{d} x$;
(11) $\int_{\frac{1}{c}}^{\mathrm{r}}|\ln x| \mathrm{d} x$;
$(12) \int_{11}^{1}\left(1-x^{2}\right)^{\frac{m}{2}} \mathrm{~d} x \quad\left(m \in \mathrm{N}^{+}\right)$;
(13) $J_{m}=\int_{0}^{\pi} x \sin ^{m} x \mathrm{~d} x \quad\left(m \in N^{*}\right)$.

## 第四节 反常积分

在一些实际问题中, 常会遇到积分区间为无穷区间,或者被积函数为无界函 数的积分,它们已经不属于前面所说的定积分了.因此, 我们对定积分作如下两 种推广, 从而形成反常积分的概念.

## 一、无穷限的反常积分

定义 1 设函数 $f(x)$ 在区间 $[a,+\infty)$ 上连续, 取 $\iota>a$, 如果极限

$$
\lim _{x \rightarrow+\infty} \int_{a}^{\prime} f(x) \mathrm{d} x
$$

存在, 则称此极限为函数 $f(x)$ 在无穷区间 $[a,+\infty)$ 上的反常积分, 记作 $\int_{a}^{+\infty} f(x) \mathrm{d} x$, 即

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x=\lim _{t \rightarrow+\infty} \int_{a}^{t} f(x) \mathrm{d} x,
$$

这时也称反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 悠敛; 如果上述极限不存在, 则函数 $f(x)$ 在无穷 区间 $[a,+\infty)$ 上的反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 就没有意义, 习惯上称为反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散,这时记号 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 不再表示数值了.

类似地, 设函数 $f(x)$ 在区间 $(-\infty, b]$ 上连续, 取 $t<b$. 如果极限

$$
\lim _{x \rightarrow-\infty} \int_{1}^{b} f(x) \mathrm{d} x
$$

存在, 则称此极限为函数 $f(x)$ 在无穷区间 $(-\infty, b]$ 上的反常积分, 记作 $\int_{-\infty}^{b} f(x) \mathrm{d} x$, 即

$$
\int_{-\infty}^{b} f(x) \mathrm{d} x=\lim _{b \rightarrow-\infty} \int_{1}^{b} f(x) \mathrm{d} x,
$$

这时也称反常积分 $\int_{-\infty}^{b} f(x) \mathrm{d} x$ 愍㧱; 如果上述极限不存在, 则称反常积分 $\int_{-\infty}^{b} f(x) \mathrm{d} x$ 发散 .

设函数 $f(x)$ 在区间 $(-\infty,+\infty)$ 上连续, 如果反常积分

$$
\int_{-\infty}^{10} f(x) \mathrm{d} x \text { 和 } \int_{0}^{+\infty} f(x) \mathrm{d} x
$$

都收敛, 则称上述两反常积分之和为函数 $f(x)$ 在无穷区间 $(-\infty,+\infty)$ 上的反 常积分，记作 $\int_{-\infty}^{+\infty} f(x) \mathrm{d} x$ ，即

$$
\begin{aligned}
\int_{-\infty}^{+\infty} f(x) \mathrm{d} x & =\int_{-\infty}^{+1} f(x) \mathrm{d} x+\int_{0}^{+\infty} f(x) \mathrm{d} x \\
& =\lim _{1 \rightarrow-\infty} \int_{t}^{0} f(x) \mathrm{d} x+\lim _{t \rightarrow+\infty} \int_{0}^{1} f(x) \mathrm{d} x,
\end{aligned}
$$

这时也称反常积分 $\int_{-\infty}^{+\infty} f(x) \mathrm{d} x$ 悠酶; 否则就称反常积分 $\int_{-\infty}^{+\infty} f(x) \mathrm{d} x$ 发散.

上述反常积分统称为无极限的反常积分.

由上述定义及牛顿 - 莱布尼茨公式, 可得如下结果.

设 $F(x)$ 为 $f(x)$ 在 $[a,+\infty)$ 上的一个原函数, 若 $\lim _{x \rightarrow+\infty} F(x)$ 存在, 则反常 积分

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x=\lim _{x \rightarrow+\infty} F(x)-F(a) ;
$$

若 $\lim _{x \rightarrow+\infty} F(x)$ 不存在,则反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散.

如果记 $F(+\infty)=\lim _{x \rightarrow+\infty} F(x),[F(x)]_{1,}{ }^{\infty}=F(+\infty)-F(a)$, 则当 $F(+\infty)$ 存在时,

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x=[F(x)]_{a}^{+\infty} ;
$$

当 $F(+\infty)$ 不存在时, 反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散.

类似地, 若在 $(-\infty, b]$ 上 $F^{\prime}(x)=f(x)$, 则当 $F(-\infty)$ 存在时,

$$
\int_{-\infty}^{b} f(x) \mathrm{d} x=[F(x)]_{-\infty}^{\prime \prime} ;
$$

当 $F(-\infty)$ 不存在时, 反常积分 $\int_{-\infty}^{\prime \prime} f(x) \mathrm{d} x$ 发散.

若在 $(-\infty,+\infty)$ 内 $F^{\prime}(x)=f(x)$, 则当 $F(-\infty)$ 与 $F(+\infty)$ 都存在时,

$$
\int_{-\infty}^{1 \infty} f(x) \mathrm{d} \cdot x=[F(x)]_{-\infty}^{+\infty} ;
$$

当 $F(-\infty)$ 与 $F(+\infty)$ 有一个不存在时, 反常积分 $\int_{\alpha}^{+\infty} f(x) \mathrm{d} x$ 发散.

例 1 计算反常积分 $\int_{-\infty}^{+\infty} \frac{\mathrm{d} x}{1+x^{2}}$.

解 $\int_{-\infty}^{+\infty} \frac{\mathrm{d} x}{1+x^{2}}=[\arctan x] !_{-\infty}^{\infty}$

$$
\begin{aligned}
& =\lim _{x \rightarrow \infty} \arctan x-\lim _{x \rightarrow \infty} \arctan x \\
& =\frac{\pi}{2}-\left(-\frac{\pi}{2}\right)=\pi .
\end{aligned}
$$

这个反常积分值的几何意义是: 当 $a \rightarrow-\infty 、 b \rightarrow+\infty$ 时, 虽然图 5-8 中阴影 部分向左、在无限延伸, 但其面积却有极 限值 $\pi$. 简单地说, 它是位于曲线 $y=$ $\frac{1}{1+x^{2}}$ 的下方, $x$ 轴上方的图形面积.

例 2 计算反常积分 $\int_{1}^{+\infty} t e^{m} \mathrm{~d} t$, 其

$$
\text { 解 } \begin{aligned}
\int_{0}^{+\infty} t \mathrm{e}^{-p t} \mathrm{~d} t & =\left[\int t \mathrm{e}^{-\mu t} \mathrm{~d} t\right]_{0}^{+\infty}=\left[-\frac{1}{p} \int t \mathrm{~d}\left(\mathrm{e}^{-p}\right)\right]_{0}^{+\infty} \\
& =\left[-\frac{t}{p} \mathrm{e}^{-p t}+\frac{1}{p} \int \mathrm{e}^{-p t} \mathrm{~d} t\right]_{11}^{1 \infty} \\
& =\left[-\frac{t}{p} \mathrm{e}^{-p t}\right]_{0}^{+\infty}-\left[\frac{1}{p^{2}} \mathrm{e}^{-p t}\right]_{0}^{+\infty} \\
& =-\frac{1}{p} \lim _{1 \rightarrow+\infty} t \mathrm{e}^{-p t}-0-\frac{1}{p^{2}}(0-1)=\frac{1}{p^{2}} .
\end{aligned}
$$

注意, 上式中的极限 $\lim _{i \rightarrow+\infty} t e^{\cdots}$ 是未定式, 可用洛必达法则确定.

例 3 证明反常积分 $\int_{a}^{+\infty} \frac{\mathrm{d} x}{x^{p}}(a>0)$ 当 $p>1$ 时收敛,当 $p \leqslant 1$ 时发散.

证 当 $p=1$ 时,

$$
\int_{a}^{+\infty} \frac{\mathrm{d} x}{x^{p}}=\int_{a}^{+\infty} \frac{\mathrm{d} x}{x}=[\ln x]_{a}^{+\infty}=+\infty,
$$

当 $p \neq 1$ 时,

$$
\int_{a}^{1 \infty} \frac{\mathrm{d} x}{x^{p}}=\left[\frac{x^{1} p}{1-p}\right]_{a}^{+\infty}=\left\{\begin{array}{l}
+\infty, p<1, \\
\frac{a^{1-p}}{p-1}, p>1 .
\end{array}\right.
$$

因此, 当 $p>1$ 时, 这反常积分收敛, 其值为 $\frac{a^{1 \cdot p}}{p-1}$; 当 $p \leqslant 1$ 时, 这反常积分发 散.

## 二、无界函数的反常积分

现在我们把定积分推广到被积函数为无界函数的情形.

如果函数 $f(x)$ 在点 $a$ 的任一邻域内都无界, 那么点 $a$ 称为函数 $f(x)$ 的 断点(也称为无界间断点). 无界函数的反常积分又称为瑕积分.

定义 2 设函数 $f(x)$ 在 $(a, b]$ 上连续, 点 $a$ 为 $f(x)$ 的取点. 取 $t>a$, 如果 极限

$$
\lim _{1 \rightarrow a} \int_{1}^{b} f(x) \mathrm{d} x
$$

存在, 则称此极限为函数 $f(x)$ 在 $(a, b]$ 上的反常积分, 仍然记作 $\int_{a}^{b} f(x) \mathrm{d} x$, 即

$$
\int_{a}^{h} f(x) \mathrm{d} x=\lim _{1 \rightarrow u^{+}} \int_{1}^{h} f(x) \mathrm{d} x .
$$

这时也称反常积分 $\int_{u}^{h} f(x) \mathrm{d} x$ 悠敛. 如果上述极限不存在, 则称反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 发散.

类似地, 设函数 $f(x)$ 在 $[a, b)$ 上连续, 点 $b$ 为 $f(x)$ 的层点. 取 $t<b$, 如果 极限

存在, 则定义

$$
\lim _{1 \rightarrow b^{-}} \int_{a}^{l} f(x) \mathrm{d} x
$$

$$
\int_{a}^{b} f(x) \mathrm{d} x=\lim _{a \rightarrow i^{-}} \int_{a}^{l} f(x) \mathrm{d} x ;
$$

否则, 就称反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 发散.

设函数 $f(x)$ 在 $[a, b]$ 上除点 $c \quad(a<c<b)$ 外连续, 点 $c$ 为 $f(x)$ 的瑕点. 如 果两个反常积分

都收敛, 则定义

$$
\int_{a}^{c} f(x) \mathrm{d} x \text { 与 } \int_{c}^{b} f(x) \mathrm{d} x
$$

$$
\begin{aligned}
\int_{a}^{a} f(x) \mathrm{d} x & =\int_{a}^{b} f(x) \mathrm{d} x+\int_{c}^{b} f(x) \mathrm{d} x \\
& =\lim _{r \rightarrow c^{-}} \int_{a}^{b} f(x) \mathrm{d} x+\lim _{i \rightarrow c^{+}} \int_{i}^{b} f(x) \mathrm{d} x ;
\end{aligned}
$$

否则, 就称反常积分 $\int_{a}^{t} f(x) \mathrm{d} x$ 发散.

计算无界函数的反常积分, 也可借助于牛顿一莱布尼茨公式.

设 $x=a$ 为 $f(x)$ 的珢点, 在 $(a, b]$ 上 $F^{\prime}(x)=f(x)$, 如果极限 $\lim _{x \rightarrow a^{\prime}} F(x)$ 存 在, 则反常积分

$$
\int_{a}^{b} f(x) \mathrm{d} x=F(b)-\lim _{x \rightarrow a^{+}} F(x)=F(b)-F\left(a^{+}\right) ;
$$

如果 $\lim _{x \rightarrow a^{+}} F(x)$ 不存在, 则反常积分 $\int_{a}^{h} f(x) \mathrm{d} x$ 发散.

我们仍用记号 $[F(x)]_{a}^{b}$ 来表示 $F(b)-F\left(a^{\prime}\right)$, 从而形式上仍有

$$
\int_{a}^{\prime \prime} f(x) \mathrm{d} x=[F(x)]_{a}^{\prime \prime} .
$$

对于 $f(x)$ 在 $[a, b)$ 上连续, $b$ 为瑕点的反常积分, 也有类似的计算公式, 这 里不再详述.

例 4 计算反常积分

$$
\int_{a}^{a} \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}}(a>0)
$$

解 因为

$$
\lim _{x \rightarrow a^{-}} \frac{1}{\sqrt{a^{2}-x^{2}}}=+\infty
$$

所以点 $a$ 是瑕点,于是

$$
\int_{0}^{a} \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}}=\left[\arcsin \frac{x}{a}\right]_{11}^{a}=\lim _{x \rightarrow a^{-}} \arcsin \frac{x}{a}-0=\frac{\pi}{2} .
$$

这个反常积分值的几何意义是: 位于曲线 $y=$ $\frac{1}{\sqrt{a^{2}-x^{2}}}$ 之下, $x$ 轴之上, 直线 $x=0$ 与 $x=a$ 之间的 图形面积 (图 5-9).

例 5 讨论反常积分 $\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}}$ 的收敛性.

解 被积函数 $f(x)=\frac{1}{x^{2}}$ 在积分区间 $[-1,1]$ 上 除 $x=0$ 外连续, 且 $\lim _{x \rightarrow 0} \frac{1}{x^{2}}=\infty$.

由于

$$
\int_{-1}^{0} \frac{\mathrm{d} x}{x^{2}}=\left[-\frac{1}{x}\right]_{-1}^{0}=\lim _{x \rightarrow 1^{-}}\left(-\frac{1}{x}\right)-1=+\infty,
$$

即反常积分 $\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}}$ 发散, 所以反常积分 $\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}}$ 发散.

注意 如果疏忽了 $\dot{x}=0$ 是被积函数的瑕点, 就会得到以下的错误结果：

$$
\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}}=\left[-\frac{1}{x}\right]_{-1}^{1}=-1-1=-2 \text {. }
$$

例 6 证明反常积分 $\int_{a}^{\prime \prime} \frac{\mathrm{d} x}{(x-a)^{4}}$ 当 $0<q<1$ 时收敛; 当 $q \geqslant 1$ 时发散. 证 当 $q=1$ 时,

$$
\begin{aligned}
\int_{a}^{a} \frac{\mathrm{d} x}{(x-a)^{4}} & =\int_{a}^{b} \frac{\mathrm{d} x}{x-a}=[\ln (x-a)]_{a}^{a} \\
& =\ln (b-a)-\lim _{x \rightarrow a^{+}} \ln (x-a)=+\infty .
\end{aligned}
$$

当 $q \neq 1$ 时,

$$
\int_{a}^{b} \frac{\mathrm{d} x}{(x-a)^{q}}=\left[\frac{(x-a)^{1-q}}{1-q}\right]_{a}^{b}=\left\{\begin{array}{l}
\frac{(b-a)^{1-u}}{1-q}, 0<q<1, \\
+\infty, q>1 .
\end{array}\right.
$$

因此, 当 $0<q<1$ 时, 这反常积分收敛,其值为 $\frac{(b-a)^{1-4}}{1-q}$; 当 $q \geqslant 1$ 时, 这反 常积分发散。

设有反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$, 其中 $f(x)$ 在开区间 $(a, b)$ 内连续, $a$ 可以是 $-\infty, b$ 可以是 $+\infty, a 、 b$ 也可以是 $f(x)$ 的珢点. 对这样的反常积分, 在另加换 元函数单调的假定下,可以像定积分一样作换元.

例 7 求反常积分 $\int_{11}^{+\infty} \frac{\mathrm{d} x}{\sqrt{x(x+1)^{3}}}$.

解 这里, 积分上限为 $+\infty$, 且下限 $x=0$ 为被积函数的瑕点.

令 $\sqrt{x}=t$, 则 $x=t^{2}, x \rightarrow 0^{+}$吋 $t \rightarrow 0, x \rightarrow+\infty$ 时 $t \rightarrow+\infty$. 于是

$$
\int_{11}^{+\infty} \frac{\mathrm{d} x}{\sqrt{x(x+1)^{3}}}=\int_{0}^{+\infty} \frac{2 t \mathrm{~d} t}{t\left(t^{2}+1\right)^{3 / 2}}=2 \int_{11}^{+\infty} \frac{\mathrm{d} t}{\left(t^{2}+1\right)^{3 / 2}} .
$$

再令 $t=\tan u$, 取 $u=\arctan t, t=0$ 时 $u=0, t \rightarrow+\infty$ 时 $u \rightarrow \frac{\pi}{2}$. 于是

$$
\int_{0}^{+\infty} \frac{\mathrm{d} x}{\sqrt{x(x+1)^{3}}}=2 \int_{0}^{\frac{\pi}{2}} \frac{\sec ^{2} u \mathrm{~d} u}{\sec ^{3} u}=2 \int_{0}^{\frac{\pi}{2}} \cos u \mathrm{~d} u=2 .
$$

本例如用变换 $\frac{1}{x}=t$ 或 $\frac{1}{x+1}=t$, 计算会更简单些, 读者可自行解之.

## 习 题 5-4

1. 判定下列各反常积分的收敛性，如果收敛，计算反常积分的值：
(1) $\int_{1}^{+\infty} \frac{\mathrm{d} x}{x^{4}}$;
(2) $\int_{1}^{+\infty} \frac{\mathrm{d} x}{\sqrt{x}}$
(3) $\int_{0}^{+\infty} \mathrm{e}^{-a . x} \mathrm{~d} x(a>0)$;
(4) $\int_{0}^{+\infty} \frac{d \cdot x}{(1+x)\left(1+x^{2}\right)}$;
(5) $\int_{0}^{\infty} \mathrm{e}^{-\mu t} \sin \omega r \mathrm{~d} t \quad(p>0, \omega>0)$;
(6) $\int_{-\infty}^{+\infty} \frac{\mathrm{d} x}{x^{2}+2 x+2}$;
(7) $\int_{11}^{1} \frac{x \mathrm{~d} x}{\sqrt{1-x^{2}}}$;
(8) $\int_{1}^{2} \frac{\mathrm{d} x}{(1-x)^{2}}$;
(9) $\int_{1}^{2} \frac{x \mathrm{~d} x}{\sqrt{x-1}}$;
(10) $\int_{1}^{r} \frac{\mathrm{d} x}{x \sqrt{1-(\ln x)^{2}}}$.
2. 当 $k$ 为何值时, 反常积分 $\int_{2}^{100} \frac{\mathrm{d} x}{x(\ln x)^{x}}$ 收敛? 当 $k$ 为何值时, 这反常积分发散?又当 $k$ 为何值时，这反常积分取得垠小值?
3. 利用递推公式计算反常积分 $I_{n}=\int_{n}^{1+\infty} x^{n} \mathrm{e}^{-x} \mathrm{~d} x \quad(n \in \mathrm{N})$.

## *第五节 反常积分的审敛法 $\Gamma$ 函数

反常积分的收敛性,可以通过求被积函数的原函数,然后按定义取极限, 根 据极限的存在与否来判定. 本节中我们来建立不通过被积函数的原函数判定反 常积分收敛性的判定法.

## 一、无穷限反常积分的里敛法

定理 1 设函数 $f(x)$ 在区间 $[a,+\infty)$ 上连续, 且 $f(x) \geqslant 0$. 若函数

$$
F(x)=\int_{a}^{x} f(t) \mathrm{d} t
$$

在 $[a,+\infty)$ 上有上界, 则反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} . r$ 收敛.

事实上, 因为 $f(x) \geqslant 0, F(x)$ 在 $[a,+\infty)$ 上单调增加, 又 $F(x)$ 佂 $[a,+\infty)$ 上有上界, 故 $F(x)$ 在 $[a,+\infty)$ 上是单调有界的函数. 按照“ $[a,+\infty)$ 上的单调有界函数 $F(x)$ 必有极限 $\lim _{-\infty} F(x)$ ”的准则, 就可知道极限

$$
\lim _{1 \rightarrow \infty} \int_{a}^{\prime} f(t) \mathrm{d} t
$$

存在,即反常积分 $\int_{a}^{10} f(x) \mathrm{d} x$ 收敛.

根据定理 1 , 对于非负函数的无穷限的反常积分, 有以下的比较审敛原理.

定理 2 (比较审敛原理) 设函数 $f(x) 、 g(x)$ 在区间 $[a,+\infty)$ 上连续. 如果 $0 \leqslant f(x) \leqslant g(x)(a \leqslant x<+\infty)$, 并且 $\int_{a}^{+\infty} g(x) \mathrm{d} x$ 收敛, 则 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 也收 敛; 如果 $0 \leqslant g(x) \leqslant f(x) \quad(a \leqslant x<+\infty)$, 并且 $\int_{a}^{+\infty} g(x) \mathrm{d} x$ 发散, 则 $\int_{a}^{1 \infty} f(x) \mathrm{d} x$ 也发散.

证 设 $a<t<+\infty$, 由 $0 \leqslant f(x) \leqslant g(x)$ 及 $\int_{a}^{+\infty} g(x) \mathrm{d} x$ 收敛, 得

$$
\int_{a}^{\prime} f(x) \mathrm{d} x \leqslant \int_{a}^{\prime} g(x) \mathrm{d} x \leqslant \int_{a}^{1+\infty} g(x) \mathrm{d} x .
$$

这表明作为积分上限 $t$ 的函数

$$
F(t)=\int_{a}^{t} f(x) \mathrm{d} x
$$

在 $[a,+\infty)$ 上有上界. 由定理 1 即知反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 收敛. 如果 $0 \leqslant g(x) \leqslant f(x)$, 且 $\int_{a}^{+\infty} g(x) \mathrm{d} x$ 发散, 则 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 必定发散. 因为 如果 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 收敛, 由定理的第一部分即知 $\int_{a}^{+\infty} g(x) \mathrm{d} x$ 也收敛, 这与假设相 矛盾. 证毕.

由上节例 3 知道, 反常积分 $\int_{a}^{+\infty} \frac{\mathrm{d} x}{x^{p}}(a>0)$ 当 $p>1$ 时收敛; 当 $p \leqslant 1$ 时发 散. 因此,取 $g(x)=\frac{A}{x^{p}}(A>0)$, 立即可得下面的反常积分的比较审敛法.

定理 3(比较审敛法 1) 设函数 $f(x)$ 在区间 $[a,+\infty)(a>0)$ 上连续,且 $f(x) \geqslant 0$. 如果存在常数 $M>0$ 及 $p>1$, 使得 $f(x) \leqslant \frac{M}{x^{p}}(a \leqslant x<+\infty)$, 则反 常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 收敛; 如果存在常数 $N>0$, 使得 $f(x) \geqslant \frac{N}{x}(a \leqslant x<+\infty)$, 则反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散.

例 1 判定反常积分 $\int_{1}^{+\infty} \frac{\mathrm{d} x}{\sqrt[3]{x^{4}+1}}$ 的收敛性.

解 由于

$$
0<\frac{1}{\sqrt[3]{x^{4}+1}}<\frac{1}{\sqrt[3]{x^{4}}}=\frac{1}{x^{4 / 3}},
$$

根据比较审敛法 1 , 这个反常积分收敛.

以比较审敛法 1 为基础，可以得到在应用上较为方便的极限审敛法.

定理 4(极限审敛法 1) 设函数 $f(x)$ 在区间 $[a,+\infty)$ 上连续,且 $f(x) \geqslant 0$. 如果存在常数 $p>1$, 使得 $\lim _{x \rightarrow+\infty} x^{p} f(x)$ 存在, 则反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 收敛; 如果 $\lim _{x \rightarrow+\infty} x f(x)=d>0$ (或 $\lim _{x \rightarrow \infty} x f(x)=+\infty$ ), 则反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散.

证 设 $\lim _{x \rightarrow+\infty} x^{n} f(x)=c \quad(p>1)$. 根据极限的定义, 存在充分大的 $x_{1} \quad\left(x_{1}\right.$ $\left.\geqslant a, x_{1}>0\right)$, 当 $x>x_{1}$ 时, 必有

由此得

$$
\begin{gathered}
\left|x^{p} f(x)-c\right|<1, \\
0 \leqslant x^{p} f(x)<1+c .
\end{gathered}
$$

令 $1+c=M>0$, 于是在区间 $x_{1}<x<+\infty$ 内不等式 $0 \leqslant f(x)<\frac{M}{x^{p}}$ 成立. 由比 较审敛法 1 知 $\int_{x_{1}}^{+\infty} f(x) \mathrm{d} x$ 收敛, 而

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x=\lim _{t \rightarrow+\infty} \int_{a}^{t} f(x) \mathrm{d} x=\lim _{t \rightarrow+\infty}\left[\int_{a}^{x_{1}} f(x) \mathrm{d} x+\int_{r_{1}}^{t} f(x) \mathrm{d} x\right]
$$

$$
\begin{aligned}
& =\int_{a}^{x_{1}} f(x) \mathrm{d} x+\lim _{1 \rightarrow+\infty} \int_{x_{1}}^{t} f(x) \mathrm{d} x \\
& =\int_{a}^{x_{1}} f(x) \mathrm{d} x+\int_{x_{1}}^{+\infty} f(x) \mathrm{d} x,
\end{aligned}
$$

故反常积分

$$
\int_{u}^{+\infty} f(x) \mathrm{d} x
$$

收敛。

如果 $\lim _{x \rightarrow+\infty} x f(x)=d>0$ (或 $\left.+\infty\right)$, 则存在充分大的 $x_{1}$, 当 $x>x_{1}$ 时, 必有

$$
|x f(x)-d|<\frac{d}{2}
$$

由此得

$$
x f(x)>\frac{d}{2} \text {. }
$$

（当 $\lim _{x \rightarrow+\infty} x f(x)=+\infty$ 时, 可取任意正数作为 $d$.) 令 $\frac{d}{2}=N>0$. 于是在区间 $x_{1}<x<+\infty$ 内不等式 $f(x) \geqslant \frac{N}{x}$ 成立. 根据比较审敛法 1 知 $\int_{x_{1}}^{+\infty} f(x) \mathrm{d} x$ 发散, 从而反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 发散.

例 2 判定反常积分 $\int_{1}^{+\infty} \frac{\mathrm{d} x}{x \sqrt{1+x^{2}}}$ 的收敛性.

解 由于

$$
\lim _{x \rightarrow+\infty} x^{2} \cdot \frac{1}{x \sqrt{1+x^{2}}}=\lim _{x \rightarrow+\infty} \frac{1}{\sqrt{\frac{1}{x^{2}}+1}}=1,
$$

根据极限审敛法 1, 知所给反常积分收敛.

例 3 判定反常积分 $\int_{1}^{+\infty} \frac{x^{3 / 2}}{1+x^{2}} \mathrm{~d} x$ 的收敛性.

解 由于

$$
\lim _{x \rightarrow+\infty} x \frac{x^{3 / 2}}{1+x^{2}}=\lim _{i \rightarrow+\infty} \frac{x^{2} \sqrt{x}}{1+x^{2}}=+\infty,
$$

根据极限审敛法 1 , 知所给反常积分发散.

例 4 判定反常积分 $\int_{1}^{+\infty} \frac{\arctan x}{x} \mathrm{~d} x$ 的收敛性.

解 由于

$$
\lim _{x \rightarrow+\infty} x \frac{\arctan x}{x}=\lim _{x \rightarrow+\infty} \arctan x=\frac{\pi}{2},
$$

根据极限审敛法 1 , 知所给反常积分发散.

假定反常积分的被积函数在所讨论的区间上可取正值也可取负值. 对于这 类反常积分的收敛性,有如下的结论.

定理 5 设函数 $f(x)$ 在区间 $[a,+\infty)$ 上连续. 如果反常积分

$$
\int_{a}^{+\infty}|f(x)| \mathrm{d} x
$$

收敛,则反常积分

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x
$$

也收敛。

证 令 $\varphi(x)=\frac{1}{2}(f(x)+|f(x)|)$. 于是 $\varphi(x) \geqslant 0$, 且 $\varphi(x) \leqslant|f(x)|$, 而 $\int_{a}^{+\infty}|f(x)| \mathrm{d} x$ 收敛, 由比较审敛原理即知 $\int_{a}^{+\infty} \varphi(x) \mathrm{d} x$ 也收敛. 但 $f(x)=$ $2 \varphi(x)-|f(x)|$, 因此

$$
\int_{a}^{+\infty} f(x) \mathrm{d} x=2 \int_{a}^{+\infty} \varphi(x) \mathrm{d} x-\int_{a}^{+\infty}|f(x)| \mathrm{d} x .
$$

可见反常积分 $\int_{a}^{1 \infty} f(x) \mathrm{d} x$ 是两个收敛的反常积分的差,因此它是收敛的. 证毕.

通常称满足定理 5 条件的反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 为绝对收敛. 于是, 定理 5 可简单地表达为: 绝对收敛的反常积分 $\int_{a}^{+\infty} f(x) \mathrm{d} x$ 必定收敛.

例 5 判定反常积分 $\int_{a}^{+\infty} \mathrm{e}^{-a x} \sin b x \mathrm{~d} x \quad(a, b$ 都是常数, 且 $a>0)$ 的收敛 性。

解 因为 $\left|\mathrm{e}^{-\alpha x} \sin b x\right| \leqslant \mathrm{e}^{-u x}$, 而 $\int_{0}^{+\infty} \mathrm{e}^{-u x} \mathrm{~d} x$ 收敛, 根据比较审敛原理, 反常 积分 $\int_{0}^{+\infty}\left|\mathrm{e}^{-u t} \sin b x\right| \mathrm{d} x$ 收敛. 由定理 5 可知所给反常积分收敛.

## 二、无界函数的反常积分的审敛法

对于无界函数的反常积分,也有类似的审敛法.

由上节例 6 知道, 反常积分

$$
\int_{a}^{b} \frac{d x}{(x-a)^{\prime \prime}}
$$

当 $q<1$ 时收敛,当 $q \geqslant 1$ 时发散.于是,与定理 3 、定理 4 类似可得如下两个审敛 法:

定理 6(比较审敛法 2) 设函数 $f(x)$ 在区间 $(a, b]$ 上连续, 且 $f(x) \geqslant 0$, $x=a$ 为 $f(x)$ 的层点. 如果存在常数 $M>0$ 及 $q<1$, 使得

$$
f(x) \leqslant \frac{M}{(x-a)^{\prime \prime}} \quad(a<x \leqslant b),
$$

则反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 收敛; 如果存在常数 $N>0$, 使得

$$
f(x) \geqslant \frac{N}{x-a} \quad(a<x \leqslant b),
$$

则反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 发散.

定理 7(极限里敛法 2) 设函数 $f(x)$ 在区间 $(a, b]$ 上连续, 且 $f(x) \geqslant 0$, $x=a$ 为 $f(x)$ 的桭点. 如果存在常数 $0<q<1$, 使得

$$
\lim _{x \rightarrow a^{+}}(x-a)^{\prime \prime} f(x)
$$

存在, 则反常积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 收敛; 如果

$$
\lim _{x \rightarrow a^{+}}(x-a) f^{\prime}(x)=d>0 \quad\left(\text { 或 } \lim _{x \rightarrow a^{+}}(x-a) f(x)=+\infty\right),
$$

则反常积分 $\int_{a}^{\prime \prime} f(x) \mathrm{d} x$ 发散.

例 6 判定反常积分 $\int_{1}^{3} \frac{\mathrm{d} x}{\ln x}$ 的收敛性.

解 这里 $x=1$ 是被积函数的层点. 由洛必达法则知

$$
\lim _{x \rightarrow 1^{+}}(x-1) \frac{1}{\ln x}=\lim _{x \rightarrow 1^{+}} \frac{1}{\frac{1}{x}}=1>0,
$$

根据极限审敛法 2 , 所给反常积分发散.

例 7 判定椭圆积分

$$
\int_{0}^{1} \frac{\mathrm{d} x}{\sqrt{\left(1-x^{2}\right)\left(1-k^{2} x^{2}\right)}}\left(k^{2}<1\right)
$$

的收敛性.

解 这里 $x=1$ 是被积函数的瑕点. 由于

$$
\begin{aligned}
& \lim _{x \rightarrow 1^{-}}(1-x)^{\frac{1}{2}} \frac{1}{\sqrt{\left(1-x^{2}\right)\left(1-k^{2} x^{2}\right)}} \\
= & \lim _{x \rightarrow 1^{-}} \frac{1}{\sqrt{(1+x)\left(1-k^{2} x^{2}\right)}}=\frac{1}{\sqrt{2\left(1-k^{2}\right)}},
\end{aligned}
$$

根据极限审敛法 2 , 所给反常积分收敛. 对于无界函数的反常积分, 当被积函数在所讨论的区间上可取正值也可取 负值时, 有与定理 5 相类似的结论, 在此不再详述.

例 8 判定反常积分 $\int_{0}^{1} \frac{1}{\sqrt{x}} \sin \frac{1}{x} \mathrm{~d} x$ 的收敛性.

解 因为 $\left|\frac{1}{\sqrt{x}} \sin \frac{1}{x}\right| \leqslant \frac{1}{\sqrt{x}}$, 而 $\int_{0}^{1} \frac{\mathrm{d} x}{\sqrt{x}}$ 收敛, 根据比较审敛原理, 反常积分 $\int_{0}^{1}\left|\frac{1}{\sqrt{x}} \sin \frac{1}{x}\right| \mathrm{d} x$ 收敛, 从而反常积分 $\int_{0}^{1} \frac{1}{\sqrt{x}} \sin \frac{1}{x} \mathrm{~d} x$ 也收敛.

## 三、 $\Gamma$ 函数

下面介绍在理论上和应用上都有重要意义的 $\Gamma$ 函数。这函数的定义是

$$
\Gamma(s)=\int_{0}^{+\infty} \mathrm{e}^{-r} x^{x-1} \mathrm{~d} x \quad(s>0) .
$$

首先讨论 (1) 式右端积分的收敛性问题. 这个积分的积分区间为无穷, 又当 $s-1<0$ 时 $x=0$ 是被积函数的版点. 为此, 分别讨论下列两个积分

$$
I_{1}=\int_{0}^{1} \mathrm{e}^{-x} x^{2-1} \mathrm{~d} x, I_{2}=\int_{1}^{+\infty} \mathrm{e}^{-x} x^{x-1} \mathrm{~d} x
$$

的收敛性。

先讨论 $I_{1}$. 当 $s \geqslant 1$ 时, $I_{1}$ 是定积分; 当 $0<s<1$ 时, 因为

$$
\mathrm{e}^{-x} \cdot x^{\mathrm{s}-1}=\frac{1}{x^{1-s}} \cdot \frac{1}{\mathrm{e}^{x}}<\frac{1}{x^{1-s}},
$$

而 $1-s<1$, 根据比较审敛法 2 , 反常积分 $I_{1}$ 收敛.

再讨论 $I_{2}$. 因为

$$
\lim _{x \rightarrow+\infty} x^{2} \cdot\left(\mathrm{e}^{-x} x^{x-1}\right)=\lim _{x \rightarrow+\infty} \frac{x^{x+1}}{\mathrm{e}^{x}}=0,
$$

根据极限审敛法 $1, I_{2}$ 也收敛.

由以上讨论即得反常积分 $\int_{0}^{+\infty} \mathrm{e}^{-x} x^{s-1} \mathrm{~d} x$ 对 $s>0$ 均收 敛. $\Gamma$ 函数的图形如图 5-10 所示.

其次讨论 $\Gamma$ 函数的几个重要性质.

1. 递推公式 $\Gamma(s+1)=s \Gamma(s)(s>0)$.

证 应用分部积分法,有

$$
\Gamma(s+1)=\int_{0}^{+\infty} \mathrm{e}^{-x} x^{s} \mathrm{~d} x=-\int_{0}^{+\infty} x^{x} \mathrm{~d}\left(\mathrm{e}^{-x}\right)
$$

$$
\begin{aligned}
& =\left[-x^{s} \mathrm{e}^{-. x}\right]_{0}^{+\infty}+s \int_{0}^{+\infty} \mathrm{e}^{-x} x^{s-1} \mathrm{~d} x \\
& =s \Gamma(s),
\end{aligned}
$$

其中 $\lim _{x \rightarrow+\infty} x^{x} \mathrm{e}^{-x}=0$ 可由洛必达法则求得.

显然, $\Gamma(1)=\int_{0}^{+\infty} \mathrm{e}^{-x} \mathrm{~d} x=1$.

反复运用递推公式,便有

$$
\begin{aligned}
& \Gamma(2)=1 \cdot \Gamma(1)=1, \\
& \Gamma(3)=2 \cdot \Gamma(2)=2 !, \\
& \Gamma(4)=3 \cdot \Gamma(3)=3 !,
\end{aligned}
$$

一般的,对任何正整数 $n$, 有

$$
\Gamma(n+1)=n !
$$

所以,我们可以把 $\Gamma$ 函数看成是阶乘的推广.

2. 当 $s \rightarrow 0^{+}$时, $\Gamma(s) \rightarrow+\infty$.

证因为

$$
\Gamma(s)=\frac{\Gamma(s+1)}{s}, \Gamma(1)=1,
$$

所以当 $s \rightarrow 0^{+}$时, $\Gamma(s) \rightarrow+\infty$ (1).

3. $\Gamma(s) \Gamma(1-s)=\frac{\pi}{\sin \pi s}(0<s<1)$.

这个公式称为余元公式, 在此我们不作证明.

当 $s=\frac{1}{2}$ 时, 由余元公式可得

$$
\Gamma\left(\frac{1}{2}\right)=\sqrt{\pi}
$$

4. 在 $\Gamma(s)=\int_{0}^{+\infty} \mathrm{e}^{-x} x^{x-1} \mathrm{~d} x$ 中, 作代换 $x=u^{2}$, 有

$$
\Gamma(s)=2 \int_{0}^{+\infty} \mathrm{e}^{-u^{2}} u^{2 x-1} \mathrm{~d} u .
$$

再令 $2 s-1=t$ 或 $s=\frac{1+t}{2}$, 即有

$$
\int_{0}^{+\infty} \mathrm{e}^{-u^{2}} u^{\prime} \mathrm{d} u=\frac{1}{2} \Gamma\left(\frac{1+t}{2}\right)(t>-1) .
$$

上式左端是应用上常见的积分, 它的值可以通过上式用 $\Gamma$ 函数计算出来.

(1) $\Gamma$ 函数在 $s>0$ 时進续. 在(2)中, 令 $s=\frac{1}{2}$, 得

$$
2 \int_{0}^{+\infty} \mathrm{e}^{-u^{2}} \mathrm{~d} u=\Gamma\left(\frac{1}{2}\right)=\sqrt{\pi} .
$$

从而

$$
\int_{0}^{+\infty} e^{-u^{2}} \mathrm{~d} u=\frac{\sqrt{\pi}}{2} .
$$

上式左端的积分是在概率论中常用的积分.

## *习 题 5-5

1. 判定下列反常积分的收敛性:
(1) $\int_{11}^{+\infty} \frac{x^{2}}{x^{4}+x^{2}+1} \mathrm{~d} x$;
(2) $\int_{1}^{+\infty} \frac{\mathrm{d} x}{x \sqrt[3]{x^{2}+1}}$;
(3) $\int_{1}^{+\infty} \sin \frac{1}{x^{2}} \mathrm{~d} x$;
(4) $\int_{0}^{+\infty} \frac{d \cdot x}{1+x|\sin x|}$;
(5) $\int_{1}^{+\infty} \frac{x \arctan x}{1+x^{3}} \mathrm{~d} x$;
(6) $\int_{1}^{2} \frac{\mathrm{d} x}{(\ln x)^{x}}$;
(7) $\int_{11}^{1} \frac{x^{4}}{\sqrt{1-x^{4}}} \mathrm{~d} x$;
(8) $\int_{1}^{1} \frac{\mathrm{d} x}{\sqrt[3]{x^{2}-3 x+2}}$.
2. 设反常积分 $\int_{1}^{+\infty} f^{2}(x) \mathrm{d} x$ 收敛。证明反常积分 $\int_{1}^{+\infty} \frac{f(x)}{x} \mathrm{~d} x$ 绝对收敛.
3. 用 $\Gamma$ 函数表示下列积分, 并指出这些积分的收敛范围:
(1) $\int_{0}^{+\infty} \mathrm{e}^{-, t^{n}} \mathrm{~d} x \quad(n>0)$;
(2) $\int_{0}^{1}\left(\ln \frac{1}{x}\right)^{n} \mathrm{~d} x$;

4. 证明 $\Gamma\left(\frac{2 k+1}{2}\right)=\frac{1 \cdot 3 \cdot 5 \cdots \cdot(2 k-1) \sqrt{\pi}}{2^{k}}$, 其中 $k \in N^{\prime}$.
5. 证明以下各式(其中 ${ }_{n} \in \mathbf{N}^{*}$ ):
(1) $2 \cdot 4 \cdot 6 \cdot \cdots \cdot 2 n=2$ " $\Gamma(n+1)$;
(2) $1 \cdot 3 \cdot 5 \cdot \cdots \cdot(2 n-1)=\frac{\Gamma(2 n)}{2^{n-1} \Gamma(n)}$;

(3) $\sqrt{\pi} \Gamma(2 n)=2^{2 n-1} \Gamma(n) \Gamma\left(n+\frac{1}{2}\right)$ (勒让德(Legendre)倍䓬公式).

## 总习题五。

## 1. 填空:

(1) 函数 $f(x)$ 在 $[a, b]$ 上有界是 $f(x)$ 在 $[a, b]$ 上可积的条件，而 $f(x)$ 在 $[a, b]$ 上连续是 $f(x)$ 在 $[a, b]$ 上可积的 条件; (2) 对 $[a,+\infty)$ 上非负、连续的函数 $f(x)$, 它的变上限积分 $\int_{a}^{r} f(t) \mathrm{d} t$ 在 $[a,+\infty)$ 上有 界是反常积分 $\int_{0}^{+\infty} f(x) \mathrm{d} . x$ 收敛的____条件;

·（3）绝对收敛的反州积分 $\int_{0}^{\infty} f(x) \mathrm{d} x$ 一定_;

(4) 函数 $f(x)$ 在 $[a, b]$ 上有定义且 $|f(x)|$ 在 $[a, b]$ 上可积, 此时积分 $\int_{a}^{b} f(x) \mathrm{d} x$ 存在.

1. 回答下列问题:

(1) 设函数 $f(x)$ 及 $g(x)$ 在区间 $[a, b]$ 上连续。且 $f(x) \geqslant g(x)$, 那么 $\int_{a}^{b}[f(x)-$ $g(x)] \mathrm{d} x$ 在几何上表示仆么?

（2）设函数 $f(x)$ 在区间 $[a, b]$ 上连续, 且 $f(x) \geqslant 0$, 那么 $\int_{a}^{b} \pi f^{2}(x) \mathrm{d} x$ 在几何上表示什 么?

(3) 如果在时刻 $t$ 以 $\varphi(t)$ 的流些(单位时间内流过的流体的体积或质量)向一水池注水, 那么 $\int_{1_{1}}^{t_{2}} \varphi(t) \mathrm{d} t$ 表示什么?

（4）如果某国人口增长的速㨌为 $“(t)$,那么 $\int_{T_{1}}^{T_{2}} u(t) \mathrm{d} t$ 表示什么?

(5) 如果一公司经营某种产品的边际利润函数为 $P^{\prime}(x)$, 那么 $\int_{10010}^{20100} P^{\prime}(x) \mathrm{d} x$ 表示什么?

-3. 利用定积分的定义计算下列极限：
(1) $\lim _{n \rightarrow \infty} \frac{1}{n} \sum_{i=1}^{n} \sqrt{1+\frac{i}{n}}$;
(2) $\lim _{n \rightarrow \infty} \frac{1^{p}+2^{n}+\cdots+n^{p}}{n^{p+1}}(p>0)$.

4. 求下列极限:

(1) $\lim _{x \rightarrow a} \frac{x}{x-a} \int_{a}^{\prime} f(t) \mathrm{d} t$, 其中 $f(x)$ 连续; (2) $\lim _{x \rightarrow+\infty} \frac{\int_{1}^{x}(\arctan t)^{2} \mathrm{~d} t}{\sqrt{x^{2}+1}}$.

5. 下列计算是否正确, 试说明理由:

(1) $\int_{-1}^{1} \frac{\mathrm{d} x}{1+x^{2}}=-\int_{-1}^{1} \frac{\mathrm{d}\left(\frac{1}{x}\right)}{1+\left(\frac{1}{x}\right)^{2}}=\left[-\arctan \frac{1}{x}\right]_{-1}^{1}=-\frac{\pi}{2}$;

(2) 因为 $\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}+x+1}=\stackrel{r=\frac{1}{t}}{=}-\int_{-1}^{1} \frac{\mathrm{d} t}{t^{2}+t+1}$,

所以

$$
\int_{-1}^{1} \frac{\mathrm{d} x}{x^{2}+x+1}=0 \text {. }
$$

(3) $\int_{-\infty}^{+\infty} \frac{x}{1+x^{2}} \mathrm{~d} x=\lim _{1 \rightarrow+\infty} \int_{-1}^{1} \frac{x}{1+x^{2}} \mathrm{~d} x=0$.

6. 设 $x>0$. 证明 7. 设 $p>0$, 证明

$$
\int_{0}^{x} \frac{1}{1+t^{2}} \mathrm{~d} t+\int_{0}^{\frac{1}{r}} \frac{1}{1+t^{2}} \mathrm{~d} t=\frac{\pi}{2}
$$

$$
\frac{p}{p+1}<\int_{11}^{1} \frac{\mathrm{d} x}{1+x^{p}}<1 .
$$

8. 设 $f(x) 、 g(x)$ 在区间 $[a, b]$ 上均连续，证明：

(1) $\left(\int_{a}^{b} f(x) g(x) \mathrm{d} x\right)^{2} \leqslant \int_{a}^{h} f^{2}(x) \mathrm{d} x \cdot \int_{a}^{b} g^{2}(x) \mathrm{d} x \quad$ (柯西 - 施瓦茨不等式);

（2） $\left(\int_{a}^{b}[f(x)+g(x)]^{2} \mathrm{~d} x\right)^{\frac{1}{2}} \leqslant\left(\int_{a}^{b} f^{2}(x) \mathrm{d} x\right)^{\frac{1}{2}}+\left(\int_{a}^{b} g^{2}(x) \mathrm{d} x\right)^{\frac{1}{2}}$ （闵可夫斯基不 等式）.

9. 设 $f(x)$ 在区间 $[a, b]$ 上连续, 且 $f(x)>0$. 证明

$$
\int_{a}^{b} f(x) \mathrm{d} x \cdot \int_{a}^{b} \frac{\mathrm{d} x}{f(x)} \geqslant(b-a)^{2} .
$$

10. 计算下列积分:
(1) $\int_{11}^{\frac{\pi}{2}} \frac{x+\sin x}{1+\cos x} \mathrm{~d} x$;
(2) $\int_{0}^{\frac{\pi}{4}} \ln (1+\tan x) \mathrm{d} x$;
(3) $\int_{0}^{a} \frac{d x}{x+\sqrt{a^{2}-x^{2}}}(a>0)$;
(4) $\int_{0}^{\frac{\pi}{2}} \sqrt{1-\sin 2 x} \mathrm{~d} x$;
(5) $\int_{0}^{\frac{\pi}{2}} \frac{\mathrm{d} x}{1+\cos ^{2} x}$;
(6) $\int_{0}^{\pi} x \sqrt{\cos ^{2} x-\cos ^{4} x} \mathrm{~d} x$;
(7) $\int_{0}^{\pi} x^{2}|\cos x| \mathrm{d} x$;
(8) $\int_{0}^{+\infty} \frac{d . x}{e^{x+1}+e^{3-x}}$;
(9) $\int_{\frac{1}{2}}^{\frac{3}{2}} \frac{\mathrm{d} x}{\sqrt{\left|x^{2}-x\right|}}$;
(10) $\int_{0}^{x} \max \left\{t^{3}, t^{2}, 1\right\} \mathrm{d} t$.
11. 设 $f(x)$ 为连续函数, 证明

$$
\int_{1}^{x} f(t)(x-t) \mathrm{d} t=\int_{0}^{x}\left(\int_{0}^{t} f(u) \mathrm{d} u\right) \mathrm{d} t .
$$

12. 设 $f(x)$ 在区间 $[a, b]$ 上连续, 且 $f(x)>0$,

$$
F(x)=\int_{a}^{r} f(t) \mathrm{d} t+\int_{a}^{r} \frac{\mathrm{d} t}{f(t)}, x \in[a, b]
$$

证明:

(1) $F^{\prime}(x) \geqslant 2$;

(2) 方程 $F(x)=0$ 在区间 $(a, b)$ 内有且仅有一个根.

13. 求 $\int_{0}^{2} f(x-1) \mathrm{d} x$, 其中

$$
f(x)=\left\{\begin{array}{l}
\frac{1}{1+x}, x \geqslant 0, \\
\frac{1}{1+e^{x}}, x<0 .
\end{array}\right.
$$

14. 设 $f(x)$ 在区间 $[a, b]$ 上连续, $g(x)$ 在区间 $[a, b]$ 上连续且不变号. 证明至少存在一 点 $\xi \in[a, b]$ ，使下式成立

$$
\int_{a}^{b} f(x) g(x) \mathrm{d} x=f(\xi) \int_{a}^{b} g(x) \mathrm{d} x \quad \text { (积分第一中值定理). }
$$

15. 证明: $\int_{0}^{+\infty} x^{n} \mathrm{e}^{-x^{2}} \mathrm{~d} x=\frac{n-1}{2} \int_{11}^{+\infty} x^{n-2} \mathrm{e}^{-x^{2}} \mathrm{~d} x(n>1)$, 并用它证明:

-16. 判定下列反陪积分的收敛性:

$$
\int_{0}^{+\infty} x^{2 n+1} \mathrm{e}^{-x^{2}} \mathrm{~d} x=\frac{1}{2} \Gamma(n+1) \quad(n \in \mathrm{N})
$$

(1) $\int_{0}^{+\infty} \frac{\sin x}{\sqrt{x^{3}}} \mathrm{~d} x$;
(2) $\int_{2}^{+\infty} \frac{d x}{x \cdot \sqrt[3]{x^{2}-3 x+2}}$
(3) $\int_{2}^{+\infty} \frac{\cos x}{\ln x} \mathrm{~d} x$;
(4) $\int_{0}^{+\infty} \frac{\mathrm{d} x}{\sqrt[3]{x^{2}(x-1)(x-2)}}$.

17. 计算下列反常积分:
(1) $\int_{0}^{\frac{\pi}{2}} \ln \sin x \mathrm{~d} x$;
(2) $\int_{0}^{+\infty} \frac{d x}{\left(1+x^{2}\right)\left(1+x^{a}\right)}(\alpha \geqslant 0)$.

## 第六章 定积分的应用

本章中我们将应用前面学过的定积分理论来分析和解决一些几何、物理中 的问题,其目的不仅在于建立计算这些几何、物理量的公式,而且更重要的还在 于介绍运用元素法将一个量表达成为定积分的分析方法.

## 第一节 定积分的元素法

在定积分的应用中, 经常采用所谓元素法. 为了说明这种方法, 我们先回顾 一下第五章中讨论过的曲边梯形的面积问题.

设 $f(x)$ 在区间 $[a, b]$ 上连续且 $f(x) \geqslant 0$, 求以曲线 $y=f(x)$ 为曲边、底为 $[a, b]$ 的曲边梯形的面积 $A$. 把这个面积 $A$ 表示为定积分

$$
A=\int_{a}^{b} f(x) \mathrm{d} x
$$

的步骤是：

（1）用任意一组分点把区间 $[a, b]$ 分成长度为 $\Delta x_{i} \quad(i=1,2, \cdots, n)$ 的 $n$ 个小区间, 相应的把曲边梯形分成 $n$ 个窄曲边梯形, 第 $i$ 个窄曲边梯形的面积 设为 $\Delta A_{i}$,于是有

$$
A=\sum_{i=1}^{n} \Delta A_{i}
$$

（2）计算 $\Delta A_{i}$ 的近似值

$$
\Delta A_{i} \approx f\left(\xi_{i}\right) \Delta x_{i} \quad\left(x_{i-1} \leqslant \xi_{i} \leqslant x_{i}\right) ;
$$

(3) 求和, 得 $A$ 的近似值

$$
A \approx \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}
$$

（4）求极限, 得

$$
A=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}=\int_{a}^{b} f(x) \mathrm{d} x .
$$

在上述问题中我们注意到,所求星 (即面积 $A$ )与区间 $[a, b]$ 有关. 如果把区 间 $[a, b]$ 分成许多部分区间, 则所求量相应的分成许多部分量 (即 $\Delta A_{i}$ ), 而所求 量等于所有部分量之和 (即 $A=\sum_{i=1}^{n} \Delta A_{i}$ ), 这一性质称为所求量对于区间 $[a, b]$ 具有可加性. 此外, 以 $f\left(\xi_{i}\right) \Delta x_{i}$ 近似代替部分量 $\Delta A_{i}$ 时, 要求它们只相 差一个比 $\Delta x_{i}$ 高阶的无穷小, 以使和式 $\sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}$ 的极限是 $A$ 的精确值, 从 而 $A$ 可以表示为定积分:

$$
A=\int_{a}^{b} f(x) \mathrm{d} x .
$$

在引出 $A$ 的积分表达式的四个步骤中, 主要的是第二步, 这一步是要确定 $\Delta A_{i}$ 的近似值 $f\left(\xi_{i}\right) \Delta x_{i}$, 使得

$$
A=\lim _{\lambda \rightarrow 0} \sum_{i=1}^{n} f\left(\xi_{i}\right) \Delta x_{i}=\int_{a}^{b} f(x) \mathrm{d} x .
$$

在实用上,为了简便起见,省略下标 $i$, 用 $\Delta A$ 表示任一小区间 $[x, x+\mathrm{d} x]$ 上的 窄曲边梯形的面积,这样,

$$
A=\sum \Delta A .
$$

取 $[x, x+\mathrm{d} x]$ 的左端点 $x$ 为 $\xi$, 以点 $x$ 处的函数值 $f(x)$ 为高 $\mathrm{d} x$ 为底的矩形的 面积 $f(x) \mathrm{d} x$ 为 $\Delta A$ 的近似值 (如图 6-1 阴影部分所示), 即

$$
\Delta A \approx f(x) \mathrm{d} x .
$$

上式右端 $f(x) \mathrm{d} x$ 叫做面积元素, 记为 $\mathrm{d} A=f(x) \mathrm{d} x$. 于是

$$
A \approx \sum f(x) \mathrm{d} x \text {, }
$$

因此

$$
A=\lim \sum f(x) \mathrm{d} x=\int_{a}^{b} f(x) \mathrm{d} x .
$$

一般的,如果某一实际问题中的所求量 $U$ 符 合下列条件:

(1) $U$ 是与一个变量 $x$ 的变化区间 $[a, b]$ 有 关的量;

(2) $U$ 对于区间 $[a, b]$ 具有可加性, 就是说, 如果把区间 $[a, b]$ 分成许多部分区间, 则 $U$ 相应 的分成许多部分量, 而 $U$ 等于所有部分量之和;

（3）部分量 $\Delta U_{i}$ 的近似值可表示为 $f\left(\xi_{i}\right) \Delta x_{i}$; 那么就可考虑用定积分来表达这个量 $U$. 通常写 出这个量 $U$ 的积分表达式的步骤是:

6-1

1) 根据问题的具体情况, 选取一个变坥例如 $x$ 为积分变量, 并确定它的变 化区间 $[a, b]$;
2) 设想把区间 $[a, b]$ 分成 $n$ 个小区间, 取其中任一小区间并记作 $[x, x+$ $\mathrm{d} x$ ], 求出相应于这个小区间的部分量 $\Delta U$ 的近似值. 如果 $\Delta U$ 能近似地表示为 $[a, b]$ 上的一个连续函数在 $x$ 处的值 $f(x)$ 与 $\mathrm{d} x$ 的乘积 $Ф$, 就把 $f(x) \mathrm{d} x$ 称为 量 $U$ 的元素且记作 $\mathrm{d} U$, 即

$$
\mathrm{d} U=f(x) \mathrm{d} x ;
$$

3）以所求量 $U$ 的元素 $f(x) \mathrm{d} x$ 为被积表达式,在区间 $[a, b]$ 上作定积分, 得

$$
U=\int_{a}^{h} f(x) \mathrm{d} x .
$$

这就是所求墨 $U$ 的积分表达式.

这个方法通常叫做元素法.下面两节中我们将应用这个方法来讨论几何、物 理中的一些问题.

## 第二节 定积分在几何学上的应用

## 一、平面图形的面积

## 1. 直角坐标情形

在第五章中我们已经知道, 由曲线 $y=f(x)(f(x) \geqslant 0)$ 及直线 $x=\dot{a}$, $x=b(a<b)$ 与 $x$ 轴所围成的曲边梯形的面积 $A$ 是定积分

$$
A=\int_{u}^{b} f(x) \mathrm{d} x,
$$

其中被积表达式 $f(x) \mathrm{d} x$ 就是直角坐标下的面积 元素, 它表示高为 $f(x)$ 、底为 $\mathrm{d} x$ 的一个矩形 面积.

应用定积分,不但可以计算曲边梯形面积,还 可以计算一些比较复杂的平面图形的面积.

例 1 计算由两条拖物线: $y^{2}=x, y=x^{2}$ 所 围成的图形的面积.

解 这两条抛物线所围成的图形如图 6-2 所示. 为了具体定出图形的所在范围, 先求出这两

(1) 这里 $\Delta U$ 与 $f(x) \mathrm{d} x$ 相美一个比 $\mathrm{d} x$ 踇阶的无穷小. 得到两个解:

$$
\left\{\begin{array}{l}
y^{2}=x, \\
y=x^{2},
\end{array}\right.
$$

$$
x=0, y=0 \text { 及 } x=1, y=1 .
$$

即这两拖物线的交点为 $(0,0)$ 及 $(1,1)$, 从而知道这图形在直线 $x=0$ 与 $x=1$ 之 间.

取横坐标 $x$ 为积分变量,它的变化区间为 $[0,1]$. 相应于 $[0,1]$ 上的任一小 区间 $[x, x+\mathrm{d} x]$ 的窄条的面积近似于高为 $\sqrt{x}-x^{2}$ 、底为 $\mathrm{d} x$ 的窄矩形的面积, 从而得到面积元素

$$
\mathrm{d} A=\left(\sqrt{x}-x^{2}\right) \mathrm{d} x .
$$

以 $\left(\sqrt{x}-x^{2}\right) \mathrm{d} x$ 为被积表达式,在闭区间 $[0,1]$ 上作定积分, 便得所求面积为

$$
A=\int_{0}^{1}\left(\sqrt{x}-x^{2}\right) \mathrm{d} x=\left[\frac{2}{3} x^{3 / 2}-\frac{x^{3}}{3}\right]_{0}^{1}=\frac{1}{3} .
$$

例 2 计算拖物线 $y^{2}=2 x$ 与直线 $y=x-4$ 所围成的图形的面积.

解 这个图形如图 6-3 所示. 为了定出这图 形所在的范围,先求出所给拖物线和直线的交点. 解方程组

$$
\left\{\begin{array}{l}
y^{2}=2 x \\
y=x-4
\end{array}\right.
$$

得交点 $(2,-2)$ 和 $(8,4)$, 从而知道这图形在直线 $y=-2$ 及 $y=4$ 之间.

现在,选取纵坐标 $y$ 为积分变量,它的变化区 间为 $[-2,4]$ (读者可以思考一下,取横坐标 $x$ 为

$$
\mathrm{d} A=\left(y+4-\frac{1}{2} y^{2}\right) \mathrm{d} y .
$$

以 $\left(y+4-\frac{1}{2} y^{2}\right) \mathrm{d} y$ 为被积表达式, 在闭区间 $[-2,4]$ 上作定积分, 便得所求的 面积为

$$
\begin{aligned}
A & =\int_{-2}^{4}\left(y+4-\frac{1}{2} y^{2}\right) \mathrm{d} y \\
& =\left[\frac{y^{2}}{2}+4 y-\frac{y^{3}}{6}\right]_{-2}^{4} \\
& =18 .
\end{aligned}
$$

由例 2 可以看到,积分变基选得适当,可使计算方便.

例 3 求椭圆 $\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}=1$ 所围成的图形的面积.

解 这椭圆关于两坐标轴都对称（图 6-4), 所以椭圆所围成的图形的面积为

$$
A=4 A_{1} \text { ， }
$$

其中 $A_{1}$ 为该椭圆在第一象限部分与两坐标轴 所围图形的面积,因此

$$
A=4 A_{1}=4 \int_{0}^{a} y \mathrm{~d} x .
$$

利用椭圆的参数方程

$$
\mid \begin{aligned}
& x=a \cos t, \\
& y=b \sin t
\end{aligned} \quad\left(0 \leqslant t \leqslant \frac{\pi}{2}\right)
$$

应用定积分换元法, 令 $x=a \cos t$, 则

$$
y=b \sin t, \mathrm{~d} x=-a \sin t \mathrm{~d} t .
$$

当 $x$ 由 0 变到 $a$ 时, $t$ 由 $\frac{\pi}{2}$ 变到 0 , 所以

$$
\begin{aligned}
A & =4 \int_{\pi / 2}^{11} b \sin t(-a \sin t) \mathrm{d} t=-4 a b \int_{\pi / 2}^{\pi} \sin ^{2} t \mathrm{~d} t \\
& =4 a b \int_{0}^{\pi / 2} \sin ^{2} t \mathrm{~d} t=4 a b \cdot \frac{1}{2} \cdot \frac{\pi}{2}=\pi a b .
\end{aligned}
$$

当 $a=b$ 时, 就得到大家所熟悉的圆面积的公式 $A=\pi a^{2}$.

## 2. 极坐标情形

某些平面图形,用极坐标来计算它们的面积比较方便.

设由曲线 $\rho=\varphi(\theta)$ 及射线 $\theta=\alpha, \theta=\beta$ 围成一图形 (简称为曲边扇形), 现在要计算 它的面积 (图 6-5). 这里, $\varphi(\theta)$ 在 $[\alpha, \beta]$ 上连 续, 且 $\varphi(\theta) \geqslant 0$.

由于当 $\theta$ 在 $[\alpha, \beta]$ 上变动时, 极径 $\rho=\varphi(\theta)$ 也随之变动, 因此所求图形的面积不能直接 利用扇形面积的公式 $A=\frac{1}{2} R^{2} \theta$ 来计算.

取极角 $\theta$ 为积分变量,它的变化区间为 $[\alpha, \beta]$. 相应于任一小区间 $[\theta, \theta+\mathrm{d} \theta]$ 的窄曲边扇形的面积可以用半径为 $\rho=\varphi(\theta)$ 、中心角为 $\mathrm{d} \theta$ 的扇形的面积来近 似代替, 从而得到这窄曲边扇形面积的近似值, 即曲边扇形的面积元素

$$
\mathrm{d} A=\frac{1}{2}[\varphi(\theta)]^{2} \mathrm{~d} \theta
$$

以 $\frac{1}{2}[\varphi(\theta)]^{2} \mathrm{~d} \theta$ 为被积表达式,在闭区间 $[\alpha, \beta]$ 上作定积分, 便得所求曲边扇形 的面积为

$$
A=\int_{a}^{\beta} \frac{1}{2}[\varphi(\theta)]^{2} \mathrm{~d} \theta .
$$

例 4 计算阿基米德螺线

$$
\rho=a \theta \quad(a>0)
$$

上相应于 $\theta$ 从 0 变到 $2 \pi$ 的一段弧与极轴所围成的图形 (图 6-6)的面积.

解 在指定的这段螺线上, $\theta$ 的变化区间为 $[0,2 \pi]$. 相应于 $[0,2 \pi]$ 上任一小 区间 $[\theta, \theta+\mathrm{d} \theta]$ 的窄曲边扇形的面积近似于半径 为 $a \theta$ 、中心角为 $\mathrm{d} \theta$ 的圆扇形的面积. 从而得到面 积元素

$$
\mathrm{d} A=\frac{1}{2}(a \theta)^{2} \mathrm{~d} \theta .
$$

于是所求面积为

$$
A=\int_{0}^{2 \pi} \frac{a^{2}}{2} \theta^{2} \mathrm{~d} \theta=\frac{a^{2}}{2}\left[\frac{\theta^{3}}{3}\right]_{0}^{2 \pi}=\frac{4}{3} a^{2} \pi^{3} .
$$

6-6

例 5 计算心形线

$$
\rho=a(1+\cos \theta) \quad(a>0)
$$

所围成的图形的面积.

解 心形线所围成的图形如图 6-7 所示. 这个图形对称于极轴, 因此所求 图形的面积 $A$ 是极轴以上部分图形面积 $A_{1}$ 的两倍.

对于极轴以上部分的图形, $\theta$ 的变化区间为 $[0, \pi]$. 相应于 $[0, \pi]$ 上任一小区间 $[\theta, \theta+\mathrm{d} \theta]$ 的窄曲边扇形的面 积近似于半径为 $a(1+\cos \theta)$ 、中心角为 $\mathrm{d} \theta$ 的圆扇形的 面积. 从而得到面积元絜

$$
\mathrm{d} A=\frac{1}{2} a^{2}(1+\cos \theta)^{2} \mathrm{~d} \theta
$$

于是

6-7

$$
\begin{aligned}
A_{1} & =\int_{0}^{\pi} \frac{1}{2} a^{2}(1+\cos \theta)^{2} \mathrm{~d} \theta=\frac{a^{2}}{2} \int_{0}^{\pi}\left(1+2 \cos \theta+\cos ^{2} \theta\right) \mathrm{d} \theta \\
& =\frac{a^{2}}{2} \int_{0}^{\pi}\left(\frac{3}{2}+2 \cos \theta+\frac{1}{2} \cos 2 \theta\right) \mathrm{d} \theta
\end{aligned}
$$

$$
=\frac{a^{2}}{2}\left[\frac{3}{2} \theta+2 \sin \theta+\frac{1}{4} \sin 2 \theta\right]_{0}^{\pi}=\frac{3}{4} \pi a^{2},
$$

因而所求面积为

$$
A=2 A_{1}=\frac{3}{2} \pi a^{2}
$$

## 二、体积

## 1. 旋转体的体积

旋转体就是由一个平面图形绕这平面内一条直线旋转一周而成的立体. 这 直线叫做旋转轴. 圆柱、圆锥、圆台、球体可以分别看成是由矩形绕它的一条边、 直角三角形绕它的直角边、直角梯形绕它的直角 腰、半圆绕它的直径旋转一周而成的立体, 所以它 们都是旋转体.

上述旋转体都可以看作是由连续曲线 $y=f(x)$ 、直线 $x=a 、 x=b$ 及 $x$ 轴所围成的曲边 梯形绕 $x$ 轴旋转一周而成的立体. 现在我们考虑用 定积分来计算这种旋转体的体积.

取横坐标 $x$ 为积分变量, 它的变化区间为 $[a, b]$. 相应于 $[a, b]$ 上的任一小区间 $[x, x+\mathrm{d} x]$

国 $6-8$ 的窄曲边梯形绕 $x$ 轴旋转而成的薄片的体积近似于以 $f(x)$ 为底半径、 $\mathrm{d} x$ 为高 的扁圆柱体的体积 (图 6-8), 即体积元素

$$
\mathrm{d} V=\pi[f(x)]^{2} \mathrm{~d} x .
$$

以 $\pi[f(x)]^{2} \mathrm{~d} x$ 为被积表达式, 在闭区间 $[a, b]$ 上作定积分, 便得所求旋转体体 积为

$$
V=\int_{a}^{b} \pi[f(x)]^{2} \mathrm{~d} x .
$$

例 6 连接坐标原点 $O$ 及点 $P(h, r)$ 的直线、直线 $x=h$ 及 $x$ 轴围成一个直 角三角形 (图 6-9). 将它绕 $x$ 轴旋转一周构成一个底半径为 $r$, 高为 $h$ 的圆锥 体.计算这圆锥体的体积.

解 过原点 $O$ 及点 $P(h, r)$ 的直线方程为

$$
y=\frac{r}{h} x .
$$

取横坐标 $x$ 为积分变量, 它的变化区间为 $[0, h]$. 圆锥体中相应于 $[0, h]$ 上 任一小区间 $[x, x+\mathrm{d} x]$ 的薄片的体积近似 于底半径为 $\frac{r}{h} x$ 、高为 $\mathrm{d} x$ 的扁圆柱体的体 积, 即体积元素

$$
\mathrm{d} V=\pi\left[\frac{r}{h} x\right]^{2} \mathrm{~d} x .
$$

于是所求圆锥体的体积为

$$
V=\int_{0}^{h} \pi\left(\frac{r}{h} x\right)^{2} \mathrm{~d} x=\frac{\pi r^{2}}{h^{2}}\left[\frac{x^{3}}{3}\right]_{0}^{h}=\frac{\pi r^{2} h}{3} .
$$

例 7 计算由椭圆

$$
\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}=1
$$

困 6-9

所围成的图形绕 $x$ 轴旋转一周而成的旋转体 (叫做旋转椭球体) 的体积.

解 这个旋转椭球体也可以看作是由半个椭圆

$$
y=\frac{b}{a} \sqrt{a^{2}-x^{2}}
$$

及 $x$ 轴围成的图形绕 $x$ 轴旋转一周而成的立体.

取 $x$ 为积分变量,它的变化区间为 $[-a, a]$. 旋转椭球体中相应于 $[-a, a]$ 上任一小区间 $[x, x+\mathrm{d} x]$ 的薄片的体积, 近 似于底半径为 $\frac{b}{a} \sqrt{a^{2}-x^{2}}$ 、高为 $\mathrm{d} x$ 的扁圆 柱体的体积 (图 6-10), 即体积元素

$$
\mathrm{d} V=\frac{\pi b^{2}}{a^{2}}\left(a^{2}-x^{2}\right) \mathrm{d} x \text {. }
$$

于是所求旋转椭球体的体积为

$$
\begin{aligned}
V & =\int_{-a}^{a} \pi \frac{b^{2}}{a^{2}}\left(a^{2}-x^{2}\right) \mathrm{d} x \\
& =\pi \frac{b^{2}}{a^{2}}\left[a^{2} x-\frac{x^{3}}{3}\right]_{-a}^{a}=\frac{4}{3} \pi a b^{2} .
\end{aligned}
$$

当 $a=b$ 时,旋转椭球体就成为半径为 $a$ 的球体,它的体积为 $\frac{4}{3} \pi a^{3}$.

用与上面类似的方法可以推出: 由曲线 $x=\varphi(y)$ 、直线 $y=c 、 y=d(c<$ $d$ )与 $y$ 轴所围成的曲边梯形, 绕 $y$ 轴旋转一周而成的旋转体 (图 6-11) 的体积 为

$$
V=\pi \int_{c}^{d}[\varphi(y)]^{2} \mathrm{~d} y
$$

6-11

例 8 计算由摆线 $x=a(t-\sin t), y=a(1-\cos t)$ 相应于 $0 \leqslant t \leqslant 2 \pi$ 的一 拱,直线 $y=0$ 所围成的图形分别绕 $x$ 轴、 $y$ 轴旋转而成的旋转体的体积.

解 按旋转体的体积公式, 所述图形绕 $x$ 轴旋转而成的旋转体的体积为

$$
\begin{aligned}
V_{x} & =\int_{0}^{2 \pi a} \pi y^{2}(x) \mathrm{d} x=\pi \int_{0}^{2 \pi} a^{2}(1-\cos t)^{2} \cdot a(1-\cos t) \mathrm{d} t \\
& =\pi a^{3} \int_{0}^{2 \pi}\left(1-3 \cos t+3 \cos ^{2} t-\cos ^{3} t\right) \mathrm{d} t=5 \pi^{2} a^{3} .
\end{aligned}
$$

所述图形绕 $y$ 轴旋转而成的旋转体的体积可看成平面图形 $O A B C$ 与 $O B C$ (图6-12)分别绕 $y$ 轴旋转而成的旋转侑的体积之差. 因此所求的体积为

$$
\begin{aligned}
V_{y} & =\int_{0}^{2 a} \pi x_{2}^{2}(y) \mathrm{d} y-\int_{0}^{2 a} \pi x_{1}^{2}(y) \mathrm{d} y \\
& =\pi \int_{2 \pi}^{\pi} a^{2}(\iota-\sin t)^{2} \cdot a \sin t \mathrm{~d} t-\pi \int_{0}^{\pi} a^{2}(t-\sin t)^{2} \cdot a \sin t \mathrm{~d} t \\
\vdots & =-\pi a^{3} \int_{0}^{2 \pi}(\iota-\sin t)^{2} \sin t \mathrm{~d} t=6 \pi^{3} a^{3} .
\end{aligned}
$$

## 2. 平行截面面积为已知的立体的体积

从计算旋转体体积的过程中可以看出: 如果一个立体不是旋转体, 但却知道 该立体上垂直于一定轴的各个截面的面积,那么,这个立体的体积也可以用定积 分来计算.

如图 6-13 所示, 取上述定轴为 $x$ 轴, 并设该立体在过点 $x=a 、 x=b$ 且垂 直于 $x$ 轴的两个平面之间. 以 $A(x)$ 表示过点 $x$ 且垂直于 $x$ 轴的截面面积. 假定 $A(x)$ 为 $x$ 的已知的连续函数. 这时, 取 $x$ 为积分变量, 它的变化区间为 $[a, b]$; 立体中相应于 $[a, b]$ 上任一小区间 $[x, x+\mathrm{d} x]$ 的一薄片的体积, 近似于底面积 为 $A(x)$ 、高为 $\mathrm{d} x$ 的扇柱体的体积, 即体积元素

$$
\mathrm{d} V=A(x) \mathrm{d} x \text {. }
$$

以 $A(x) \mathrm{d} x$ 为被积表达式,在闭区间 $[a, b]$ 上作定积分,便得所求立体的体积

$$
V=\int_{a}^{b} A(x) \mathrm{d} x .
$$

例 9 一平面经过半径为 $R$ 的圆柱体的底圆中心, 并与底面交成角 $\alpha$ (图 6-14). 计算这平面截圆柱体所得立体的体积.

解 取这平面与圆柱体的底面的交线为 $x$ 轴,底面上过圆中心、且垂直于 $x$ 轴的直线为 $y$ 轴. 那么, 底圆的方程为 $x^{2}+y^{2}=R^{2}$. 立体中过 $x$ 轴上的点 $x$ 且垂直于 $x$ 轴的截面是一个直角三角形. 它的两条直角边的长分别为 $y$ 及 $y \tan \alpha$, 即 $\sqrt{R^{2}-x^{2}}$ 及 $\sqrt{R^{2}-x^{2}} \tan \alpha$. 因而截面积为 $A(x)=\frac{1}{2}\left(R^{2}-x^{2}\right) \tan \alpha$, 于是所求立体体积为

- $V=\int_{-R}^{R} \frac{1}{2}\left(R^{2}-x^{2}\right) \tan \alpha \mathrm{d} x=\frac{1}{2} \tan \alpha\left[R^{2} x-\frac{1}{3} x^{3}\right]_{-R}^{R}$

$$
=\frac{2}{3} R^{3} \tan \alpha \text {. }
$$

例 10 求以半径为 $R$ 的圆为底、平行且等于底圆直径的线段为顶、高为 $h$ 的正擘锥体的体积.

解 取底圆所在的平面为 $x O y$ 平面, 圆心 $O$ 为原点,并使 $x$ 轴与正䢃雉的 顶平行 (图 6-15). 底圆的方程为 $x^{2}+y^{2}=R^{2}$. 过 $x$ 轴上的点 $x(-R \leqslant x \leqslant R)$ 作垂直于 $x$ 轴的平面,截正䢃雉体得等腰三角形. 这截面的面积为

$$
A(x)=h \cdot y=h \sqrt{R^{2}-x^{2}} \text {, }
$$

于是所求正䢃锥体的体积为

$$
\begin{aligned}
V & =\int_{-R}^{R} A(x) \mathrm{d} x=h \int_{-R}^{R} \sqrt{R^{2}-x^{2}} \mathrm{~d} x \\
& =2 R^{2} h \int_{0}^{\frac{\pi}{2}} \cos ^{2} \theta \mathrm{d} \theta=\frac{\pi R^{2} h}{2} .
\end{aligned}
$$

由此可知正䢃锥体的体积等于同底同高的圆柱体体积的一半.

## 三、平面曲线的弧长

我们知道, 圆的周长可以利用圆的内接正多边形的周长当边数无限增多时 的极限来确定. 现在用类似的方法来建立平面的连续曲线弧长的概念, 从而应用 定积分来计算弧长.

设 $A 、 B$ 是曲线弧的两个端点. 在弧 $\overparen{A B}$ 上依次任取分点 $A=M_{13}, M_{1}, M_{2}, \cdots$, $M_{i-1}, M_{i}, \cdots, M_{n-1}, M_{n}=B$, 并依次连接相邻的分点得 一折线 (图 6-16). 当分点的数目无限增加且每个小段 $\widetilde{M_{i-1}} M_{i}$ 都缩向一点时, 如果此折线的长 $\sum_{i=1}^{n}\left|M_{i-1} M_{i}\right|$ 的极限存在, 则称此极限为豊线弧 $\widehat{A B}$ 的弧长, 并称 此曲线弧 $\widehat{A B}$ 是可求长的.

对光滑的曲线弧 (参看第 171 页上的脚注), 有 如下结论:

## 定理 光滑曲线弧是可求长的.

这个定理我们不加证明. 由于光滑曲线弧是可求长的, 故可应用定积分来计 算弧长.下面我们利用定积分的元素法来讨论平面光滑曲线弧长的计算公式.

设曲线弧由参数方程

$$
\left\{\begin{array}{l}
x=\varphi(t), \\
y=\psi(t)
\end{array} \quad(\alpha \leqslant t \leqslant \beta)\right.
$$

给出, 其中 $\varphi(t) 、 \psi(t)$ 在 $[\alpha, \beta]$ 上具有连续导数, 且 $\varphi^{\prime}(t) 、 \psi^{\prime}(t)$ 不同时为零. 现在来计算这曲线弧的长度.

取参数 $t$ 为积分变量, 它的变化区间为 $[\alpha, \beta]$. 相应于 $[\alpha, \beta]$ 上任一小区间 $[t, t+\mathrm{d} t]$ 的小弧段的长度 $\Delta s$ 近似等于对应的弦的长度 $\sqrt{(\Delta x)^{2}+(\Delta y)^{2}}$, 因为

$$
\begin{aligned}
& \Delta x=\varphi(t+\mathrm{d} t)-\varphi(t) \approx \mathrm{d} x=\varphi^{\prime}(t) \mathrm{d} t, \\
& \Delta y=\psi(t+\mathrm{d} t)-\psi(t) \approx \mathrm{d} y=\psi^{\prime}(t) \mathrm{d} t,
\end{aligned}
$$

所以, $\Delta s$ 的近似值 (弧微分) 即弧长元素为

$$
\mathrm{d} s=\sqrt{(\mathrm{d} x)^{2}+(\mathrm{d} y)^{2}}=\sqrt{\varphi^{\prime 2}(t)(\mathrm{d} t)^{2}+\psi^{\prime 2}(t)(\mathrm{d} t)^{2}}
$$

$$
=\sqrt{\varphi^{\prime 2}(t)+\psi^{\prime 2}(t)} \mathrm{d} t .
$$

于是所求弧长为

$$
s=\int_{a}^{\beta} \sqrt{\varphi^{\prime 2}(t)+\psi^{\prime 2}(t)} \mathrm{d} t .
$$

当曲线弧由直角坐标方程

$$
y=f(x) \quad(a \leqslant x \leqslant b)
$$

给出, 其中 $f(x)$ 在 $[a, b]$ 上具有一阶连续导数, 这时曲线弧有参数方程

$$
\mid \begin{aligned}
& x=x, \\
& y=f(x)
\end{aligned} \quad(a \leqslant x \leqslant b),
$$

从而所求的弧长为

$$
s=\int_{a}^{\prime \prime} \sqrt{1+y^{\prime 2}} \mathrm{~d} x .
$$

当曲线弧由极坐标方程

$$
\rho=\rho(\theta) \quad(\alpha \leqslant \theta \leqslant \beta)
$$

给出, 其中 $\rho(\theta)$ 在 $[\alpha, \beta]$ 上具有连续导数, 则由直角坐标与极坐标的关系可得

$$
\left\{\begin{array}{l}
x=\rho(\theta) \cos \theta, \\
y=\rho(\theta) \sin \theta
\end{array}(\alpha \leqslant \theta \leqslant \beta) .\right.
$$

这就是以极角 $\theta$ 为参数的曲线弧的参数方程. 于是, 弧长元素为

从而所求弧长为

$$
\mathrm{d} s=\sqrt{x^{\prime 2}(\theta)+y^{\prime 2}(\theta)} \mathrm{d} \theta=\sqrt{\rho^{2}(\theta)+\rho^{\prime 2}(\theta)} \mathrm{d} \theta,
$$

$$
s=\int_{a}^{\beta} \sqrt{\rho^{2}(\theta)+\rho^{\prime 2}(\theta)} \mathrm{d} \theta .
$$

例 11 计算曲线 $y=\frac{2}{3} x^{3 / 2}$ 上相应于 $a \leqslant x \leqslant b$ 的一段弧 (图 6-17) 的长 度。

解 因 $y^{\prime}=x^{1 / 2}$, 从而弧长元素

$$
\mathrm{d} s=\sqrt{1+\left(x^{1 / 2}\right)^{2}} \mathrm{~d} x=\sqrt{1+x} \mathrm{~d} x .
$$

因此, 所求弧长为

$$
\begin{aligned}
s & =\int_{a}^{b} \sqrt{1+x} \mathrm{~d} x=\left[\frac{2}{3}(1+x)^{3 / 2}\right]_{a}^{b} \\
& =\frac{2}{3}\left[(1+b)^{3 / 2}-(1+a)^{3 / 2}\right] .
\end{aligned}
$$

例 12 计算摆线(图 6-18)

$$
\left\{\begin{array}{l}
x=a(\theta-\sin \theta), \\
y=a(1-\cos \theta)
\end{array}\right.
$$

的一拱 $(0 \leqslant \theta \leqslant 2 \pi)$ 的长度.

解 弧长元素为

$$
\begin{aligned}
\mathrm{d} s & =\sqrt{a^{2}(1-\cos \theta)^{2}+a^{2} \sin ^{2} \theta} \mathrm{d} \theta \\
& =a \sqrt{2(1-\cos \theta)} \mathrm{d} \theta=2 a \sin \frac{\theta}{2} \mathrm{~d} \theta .
\end{aligned}
$$

从而,所求弧长

$$
s=\int_{0}^{2 \pi} 2 a \sin \frac{\theta}{2} \mathrm{~d} \theta=2 a\left[-2 \cos \frac{\theta}{2}\right]_{11}^{2 \pi}=8 a .
$$

国 6-18

例 13 求阿基米德螺线 $\rho=a \theta(a>0)$ 相应于 $0 \leqslant \theta \leqslant 2 \pi$ 一段 (图 6-19) 的弧长.

解 弧长元素为

$$
\mathrm{d} s=\sqrt{a^{2} \theta^{2}+a^{2}} \mathrm{~d} \theta=a \sqrt{1+\theta^{2}} \mathrm{~d} \theta,
$$

于是所求弧长为

$$
s=a \int_{10}^{2 \pi} \sqrt{1+\theta^{2}} \mathrm{~d} \theta=\frac{a}{2}\left[2 \pi \sqrt{1+4 \pi^{2}}+\ln \left(2 \pi+\sqrt{1+4 \pi^{2}}\right)\right] .
$$

## 习 题 6-2

1. 求图 6-20 中各画斜线部分的面积(图见下页).
2. 求由下列各组曲线所围成的图形的面积:

(1) $y=\frac{1}{2} x^{2}$ 与 $x^{2}+y^{2}=8$ (两部分都要计算);

(2) $y=\frac{1}{x}$ 与直线 $y=x$ 及 $x=2$;

(3) $y=\mathrm{e}^{\mathrm{x}}, y=\mathrm{e}^{-x}$ 与直线 $x=1$;

(4) $y=\ln x, y$ 轴与直线 $y=\ln a, y=\ln b(b>a>0)$.

3. 求拖物线 $y=-x^{2}+4 x-3$ 及其在点 $(0,-3)$ 和 $(3,0)$ 处的切线所圈成的图形的 面积.

4. 求地物线 $y^{2}=2 p x$ 及其在点 $\left(\frac{p}{2}, p\right)$ 处的法线所围成的图形的面积.

5. 求由下列各曲线所围成的图形的面积:
(1) $\rho=2 a \cos \theta$;
(2) $x=a \cos ^{3} t, y=a \sin ^{3} t$;
(3) $\rho=2 a(2+\cos \theta)$.
1. 求由摆线 $x=a(t-\sin t), y=a(1-\cos t)$ 的一拱 $(0 \leqslant t \leqslant 2 \pi)$ 与横轴所围成的图形的 面积.
2. 求对数螺线 $\rho=a \mathrm{e}^{\theta}(-\pi \leqslant \theta \leqslant \pi)$ 及射线 $\theta=\pi$ 所围成的图形的面积.
3. 求下列各曲线所围成图形的公共部分的面积:
(1) $\rho=3 \cos \theta$ 及 $\rho=1+\cos \theta$;
(2) $\rho=\sqrt{2} \sin \theta$ 及 $\rho^{2}=\cos 2 \theta$.
1. 求位于曲线 $y=\mathrm{e}^{x}$ 下方, 该曲线过原点的切线的左方以及 $x$ 轴上方之间的图形的 面积.
2.  求由拁物线 $y^{2}=4 a x$ 与过焦点的弦所围成的图形面积的垠小值.
3.  把拖物线 $y^{2}=4 a x$ 及直线 $x=x_{0} \quad\left(x_{11}>0\right)$ 所围成的图形绕 $x$ 轴旋转, 计算所得旋 转体的体积.
4.  由 $y=x^{3}, x=2, y=0$ 所围成的图形,分别绕 $x$ 轴及 $y$ 轴旋转, 计算所得两个旋转体 的体积.
5.  把星形线 $x^{2 / 3}+y^{2 / 3}=a^{2 / 3}$ 所围成的图形绕 $x$ 轴旋转(图 6-21), 计算所得旋转体的 体积. 14. 用积分方法证明图 6-22 中球缺的体积为

$$
V=\pi H^{2}\left(R-\frac{H}{3}\right) \text {. }
$$

15. 求下列已知曲线所围成的图形,按指定的轴旋转所产生的旋转体的体积:

(1) $y=x^{2}, x=y^{2}$, 绕 $y$ 轴;

(2) $y=\arcsin x, x=1, y=0$, 绕 $x$ 轴;

(3) $x^{2}+(y-5)^{2}=16$, 绕 $x$ 轴;

(4) 摆线 $x=a(t-\sin t), y=a(1-\cos t)$ 的一拱, $y=0$, 绕直线 $y=2 a$.

16. 求圆盘 $x^{2}+y^{2} \leqslant a^{2}$ 绕 $x=-b(b>a>0)$ 旋转所成旋转体的体积.
17. 设有一截雉体, 其高为 $h$, 上、下底均为椭图, 椭图 的轴长分别为 $2 a 、 2 b$ 和 $2 A 、 2 B$, 求这截锥体的体积.
18. 计算底面是半径为 $R$ 的圆, 而垂直于底面上一条 固定直径的所有截面都是等边三角形的立体体积 (图 6-23).
19. 证明: 由平面图形 $0 \leqslant a \leqslant x \leqslant b, 0 \leqslant y \leqslant f(x)$ 绕 $y$ 轴旋转所成的旋转体的体积为

$$
V=2 \pi \int_{a}^{b} x f(x) \mathrm{d} x .
$$

20. 利用题 19 的结论,计算曲线 $y=\sin x(0 \leqslant x \leqslant \pi)$

21. 计算曲线 $y=\ln x$ 上相应于 $\sqrt{3} \leqslant x \leqslant \sqrt{8}$ 的一段弧 的长度.
22. 计算曲线 $y=\frac{\sqrt{x}}{3}(3-x)$ 上相应于 $1 \leqslant x \leqslant 3$ 的一 段弧(图 6-24) 的长度.
23. 计算半立方拖物线 $y^{2}=\frac{2}{3}(x-1)^{3}$ 被拖物线

24. 计算执物线 $y^{2}=2 p x$ 从顶点到这曲线上的一点 $M(x, y)$ 的弧长.
25. 计算星形线 $x=a \cos ^{3} t, y=a \sin ^{3} t$ (图 6-25) 的全长.
26. 将绕在圆 (半径为 $a$ ) 上的细线放开拉直, 使细线与圆周始终相切 (图 6-26), 细线端 点画出的轨迹叫做圆的渐伸线, 它的方程为

$$
x=a(\cos t+t \sin t), y=a(\sin t-t \cos t) \text {. }
$$

算出这曲线上相应于 $0 \leqslant t \leqslant \pi$ 的一段弧的长度.

国 6-25

6-26

27. 在摆线 $x=a(t-\sin t), y=a(1-\cos t)$ 上求分摆线第一拱成 $1: 3$ 的点的坐标.
28. 求对数螺线 $\rho=\mathrm{e}^{a \theta}$ 相应于 $0 \leqslant \theta \leqslant \varphi$ 的一段弧长.
29. 求曲线 $\rho \theta=1$ 相应于 $\frac{3}{4} \leqslant \theta \leqslant \frac{4}{3}$ 的一段弧长.
30. 求心形线 $\rho=a(1+\cos \theta)$ 的全长.

## 第三节 定积分在物理学上的应用

## 一、变力沿直线所作的功

从物理学知道, 如果物体在作直线运动的过程中有一个不变的力 $F$ 作用在 这物体上, 且这力的方向与物体运动的方向一致, 那么, 在物体移动了距离 $s$ 时, 力 $F$ 对物体所作的功为

$$
W=F \cdot s .
$$

如果物体在运动过程中所受到的力是变化的, 这就会遇到变力对物体作功 的问题.下面通过具体例子说明如何计算变力所作的功. 例 1 把一个带电荷量 $+q$ 的点电荷放在 $r$ 轴上坐标原点 $O$ 处, 它产生一 个电场. 这个电场对周围的电荷有作用力. 由物理学知道, 如果有一个单位正电 荷放在这个电场中距离原点 $O$ 为 $r$ 的地方, 那么电场对它的作用力的大小为

$$
F=k \frac{q}{r^{2}} . \text { ( } k \text { 是常数). }
$$

见图 6-27, 当这个单位正电荷在电场中从 $r=a$ 处沿 $r$ 轴移动到 $r=b(a<b)$ 处时,计算电场力 $F$ 对它所作的功.

解 在上述移动过程中, 电场对这单位正电荷的作用力是变的. 取 $r$ 为积 分变量,它的变化区间为 $[a, b]$. 设 $[r, r+\mathrm{d} r]$ 为 $[a, b]$ 上的任一小区间. 当单位正电荷从 $r$ 移动到 $r+\mathrm{d} r$ 时, 电场力对它所作的功近似于 $\frac{k q}{r^{2}} \mathrm{~d} r$, 即功为

$$
\mathrm{d} W=\frac{k q}{r^{2}} \mathrm{~d} r
$$

于是所求的功为

$$
W=\int_{a}^{b} \frac{k q}{r^{2}} \mathrm{~d} r=k q\left[-\frac{1}{r}\right]_{a}^{l}=k q\left(\frac{1}{a}-\frac{1}{b}\right) .
$$

在计算静电场中某点的电位时, 要考虑将单位正电荷从该点处 $(r=a)$ 移到 无穷远处时电场力所作的功 $W$. 此时, 电场力对单位正电荷所作的功就是反常 积分：

$$
W=\int_{a}^{+\infty} \frac{k q}{r^{2}} \mathrm{~d} r=\left[-\frac{k q}{r}\right]_{a}^{+\infty}=\frac{k q}{a} .
$$

例 2 在底面积为 $S$ 的圆柱形容器中盛有一定量的气体. 在等温条件下, 由 于气体的膨胀,把容器中的一个活塞 (面积为 $S$ ) 从点 $a$ 处推移到点 $b$ 处 (图6-28). 计算在移动 过程中,气体压力所作的功.

解 取坐标系如图 6-28 所示. 活塞的位置 可以用坐标 $x$ 来表示. 由物理学知道,一定量的

$$
p V=k \text { 或 } p=\frac{k}{V} \text {. }
$$

因为 $V=x S$, 所以

$$
p=\frac{k}{x S} .
$$

于是, 作用在活塞上的力

$$
F=p \cdot S=\frac{k}{x S} \cdot S=\frac{k}{x} .
$$

在气体膨胀过程中,体积 $V$ 是变的, 因而 $x$ 也是变的, 所以作用在活塞上 的力也是变的.

取 $x$ 为积分变量, 它的变化区间为 $[a, b]$. 设 $[x, x+\mathrm{d} x]$ 为 $[a, b]$ 上任一小 区间, 当活塞从 $x$ 移动到 $x+\mathrm{d} x$ 时, 变力 $F$ 所作的功近似于 $\frac{k}{x} \mathrm{~d} x$, 即功元素为

$$
\mathrm{d} W=\frac{k}{x} \mathrm{~d} x
$$

于是所求的功为

$$
W=\int_{a}^{b} \frac{k}{x} \mathrm{~d} x=k[\ln x]_{a}^{b}=k \ln \frac{b}{a} .
$$

下面再举一个计算功的例子, 它虽不是一个变 力作功问题,但也可用积分来计算.

例 3 一圆柱形的贮水桶高为 $5 \mathrm{~m}$, 底圆半径 为 $3 \mathrm{~m}$, 桶内盛满了水: 试问要把桶内的水全部吸出 需作多少功?

解. 作 $x$ 轴如图 6-29 所示, 取深度 $x$ (单位 为 $\mathrm{m}$ )为积分变量, 它的变化区间为 $[0,5]$. 相应于 $[0,5]$ 上任一小区间 $[x, x+\mathrm{d} x]$ 的一薄层水的高度

$$
\mathrm{d} W=88.2 \pi x \mathrm{~d} x,
$$

此即功元素. 于是所求的功为

$$
\begin{aligned}
W & =\int_{0}^{5} 88.2 \pi x \mathrm{~d} x=88.2 \pi\left[\frac{x^{2}}{2}\right]_{0}^{5} \\
& =88.2 \pi \cdot \frac{25}{2} \approx 3462(\mathrm{~kJ}) .
\end{aligned}
$$

## 二、水压力

从物理学知道, 在水深为 $h$ 处的压强为 $p=\rho g h$, 这里 $\rho$ 是水的密度, $g$ 是重 力加速度. 如果有一面积为 $A$ 的平板水平地放置在水深为 $h$ 处, 那么, 平板一侧 所受的水压力为

$$
P=p \cdot A \text {. }
$$

如果平板铅直放置在水中, 那么, 由于水深不同的点处压强 $p$ 不相等, 平板 一侧所受的水压力就不能用上述方法计算.下面举例说明它的计算方法.

例 4 一个横放着的圆柱形水桶, 桶内盛有半桶水 (图 6-30(a)). 设桶的底 半径为 $R$, 水的密度为 $\rho$, 计算桶的一个端面上所受的压力.

解 桶的一个端面是圆片, 所以现在要计算的是当水平面通过圆心时, 铅直 放置的一个半圆片的一侧所受到的水压力.

如图 6-30(b), 在这个圆片上取过圆心且铅直向下的直线为 $x$ 轴, 过圆心 的水平线为 $y$ 轴. 对这个坐标系来讲, 所讨论的半圆的方程为 $x^{2}+y^{2}=R^{2}$ $(0 \leqslant x \leqslant R)$. 取 $x$ 为积分变量, 它的变化区间为 $[0, R]$. 设 $[x, x+\mathrm{d} x]$ 为 $[0, R]$ 上的任一小区间, 半圆片上相应于 $[x ; x+\mathrm{d} x]$ 的窄条上各点处的压强近似于 $\rho g x$, 这窄条的面积近似于 $2 \sqrt{R^{2}-x^{2}} \mathrm{~d} x$. 因此, 这窄条一侧所受水压力的近似 值, 即压力元素为

$$
\mathrm{d} P=2 \rho g x \sqrt{R^{2}-x^{2}} \mathrm{~d} x .
$$

于是所求压力为

$$
\begin{aligned}
& \begin{aligned}
P & =\int_{0}^{R} 2 \rho g x \sqrt{R^{2}-x^{2}} \mathrm{~d} x=-\rho g \int_{0}^{R}\left(R^{2}-x^{2}\right)^{1 / 2} \mathrm{~d}\left(R^{2}-x^{2}\right) \\
& =-\rho g\left[\frac{2}{3}\left(R^{2}-x^{2}\right)^{3 / 2}\right]_{0}^{R}=\frac{2 \rho g}{3} R^{3} .
\end{aligned} \\
& \text { 三、引カ }
\end{aligned}
$$

从物理学知道, 质量分别为 $m_{1} 、 m_{2}$, 相距为 $r$ 的两质点间的引力的大小为

$$
F=G \frac{m_{1} m_{2}}{r^{2}},
$$

其中 $G$ 为引力系数,引力的方向沿着两质点的连线方向.

如要计算一根细棒对一个质点的引力,那么,由于细棒上各点与该质点的距 离是变化的, 且各点对该质点的引力的方向也是变化的, 因此就不能用上述公式 来计算.下面举例说明它的计算方法.

例 5 设有一长度为 $l$ 、线密度为 $\mu$ 的均匀细直棒, 在其中垂线上距棒 $a$ 单 位处有一质量为 $m$ 的质点 $M$. 试计算该棒对质点 $M$ 的引力.

解. 取坐标系如图 6-31 所示, 使棒位于 $y$ 轴 上,质点 $M$ 位于 $x$ 轴上,棒的中点为原点 $O$. 取 $y$ 为 积分变量,它的变化区间为 $\left[-\frac{l}{2}, \frac{l}{2}\right]$. 设 $[y, y+\mathrm{d} y]$ 为 $\left[-\frac{l}{2}, \frac{l}{2}\right]$ 上任一小区间, 把细直棒上相应于 $[y, y$ $+\mathrm{d} y$ ] 的一小段近似地看成质点, 其质量为 $\mu \mathrm{d} y$, 与 $M$ 相距 $r=\sqrt{a^{2}+y^{2}}$. 因此可以按照两质点间的引力 计算公式求出这小段细直棒对质点 $M$ 的引力 $\Delta F$ 的 大小为

$$
\Delta F \approx G \frac{m \mu \mathrm{d} y}{a^{2}+y^{2}},
$$

从而求出 $\Delta F$ 在水平方向分力 $\Delta F$, 的近似值, 即细直棒对质点 $M$ 的引力在水 平方向分力 $F_{. r}$ 的元素为

$$
\mathrm{d} F_{x}=-G \frac{a m \mu \mathrm{d} y}{\left(a^{2}+y^{2}\right)^{\frac{3}{2}}} .
$$

于是得引力在水平方向分力为

$$
\begin{aligned}
F_{x^{\prime}} & =-\int_{-\frac{1}{2}}^{\frac{1}{2}} \frac{G a m \mu}{\left(a^{2}+y^{2}\right)^{\frac{3}{2}}} \mathrm{~d} y \\
& =-\frac{2 G m \mu l}{a} \cdot \frac{1}{\sqrt{4 a^{2}+l^{2}}} .
\end{aligned}
$$

由对称性知,引力在铅直方向分力为 $F_{v}=0$.

当细直棒的长度 $l$ 很大时, 可视 $l$ 趋于无穷. 此时, 引力的大小为 $\frac{2 G m \mu}{a}$, 方 向与细棒垂直且由 $M$ 指向细棒.

## $\therefore$ 习 题 $6-3$

1. 由实验知道, 弹策在拉伸过程中, 需要的力 $F$ (单位: $\mathrm{N}$ ) 与伸长些 $s$ (单位: $\mathrm{cm}$ ) 成正 比, 即

$$
F=k s \quad(k \text { 是比例常数). }
$$

如果把弹管由原长拉伸 $6 \mathrm{~cm}$, 计算所作的功.

2. 直径为 $20 \mathrm{~cm}$ 、高为 $80 \mathrm{~cm}$ 的圆筒内充满压强为 $10 \mathrm{~N} / \mathrm{cm}^{2}$ 的蒸汽. 设温度保持不变, 要使蒸汽体积缩小一半,问需要作多少功?
3. (1) 证明: 把质量为 $m$ 的物体从地球表面升高到 $h$ 处所作的功是

$$
W=\frac{m g R h}{R+h},
$$

其中 $g$ 是重力加速度, $R$ 是地球的半径;

(2) 一颗人造地球卫星的质证为 $173 \mathrm{~kg}$, 在高于地面 $630 \mathrm{~km}$ 处进人轨道. 问把这颗卫星 从地面送到 $630 \mathrm{~km}$ 的高空处, 克服地球引力要作多少功? 已知 $\mathrm{g}=9.8 \mathrm{~m} / \mathrm{s}^{2}$, 地球半径 $R=$ $6370 \mathrm{~km}$.

4. 一物体按规律 $x=c t^{3}$ 作直线运动, 介质的阻力与速度的平方成正比. 计算物体由 $x$ $=0$ 移至 $x=a$ 时, 克服介质阻力所作的功.
5. 用铁锤将一铁钉击人木板, 设木板对铁钉的阻力与铁钉击人木板的深度成正比, 在击 第一次时, 将铁钉击人木板 $1 \mathrm{~cm}$. 如果铁锤每次锤击铁钓所作的功相等, 问锤击第二次时, 铁 钉又击人多少?
6. 设一圆锥形起水池, 深 $15 \mathrm{~m}$, 口经 $20 \mathrm{~m}$, 盛满水, 今以目将水吸尽, 问要作多少功?
7. 有一闸门,它的形状和尺寸如图 6-32 所示, 水面超过门顶 $2 \mathrm{~m}$. 求闸门上所受的水 压力.
8. 酒水车上的水箱是一个横放的椭圆柱体, 尺寸如图 6-33 所示. 当水箱装满水时,计 算水箱的一个端面所受的压力.

9. 有一等腰梯形闸门, 它的两条底边各长 $10 \mathrm{~m}$ 和 $6 \mathrm{~m}$, 高为 $20 \mathrm{~m}$. 较长的底边与水面相 齐.计算闸门的一侧所受的水压力.
10. 一底为 $8 \mathrm{~cm}$ 、高为 $6 \mathrm{~cm}$ 的等喓三角形片, 铅直地沉没在水中, 顶在上, 底在下且与水 面平行, 而顶离水面 $3 \mathrm{~cm}$, 试求它每面所受的压力.
11. 设有一长度为 $l$ 、线密度为 $\mu$ 的均匀细直捧, 在与棒的一端垂直距离为 $a$ 单位处有 一质为 $m$ 的质点 $M$, 试求这细棒对质点 $M$ 的引力.
12. 设有一半径为 $R$, 中心角为 $\varphi$ 的圆弧形细棒, 其线密度为常数 $\mu$. 在圆心处有一质量 为 $m$ 的质点 $M$. 试求这细棒对质点 $M$ 的引力.

## 总习题六

1. 一金属棒长 $3 \mathrm{~m}$, 离棒左端 $x \mathrm{~m}$ 处的线密度为 $\rho(x)=\frac{1}{\sqrt{x+1}}(\mathrm{~kg} / \mathrm{m})$. 问 $x$ 为何值 时, $[0, x]$ 一段的质些为全棒质证的一半.
2. 求由曲线 $\rho=a \sin \theta, \rho=a(\cos \theta+\sin \theta)(a>0)$ 所围图形公共部分的面积.
3. 设执物线 $y=a x^{2}+b x+c$ 通过点 $(0,0)$, 且当 $x \in[0,1]$ 时, $y \geqslant 0$. 试确定 $a, b, c$ 的 值, 使得扡物线 $y=a x^{2}+b x+c$ 与直线 $x=1, y=0$ 所围图形的面积为 $\frac{4}{9}$, 且使该图形绕 $x$ 轴旋转而成的旋转体的体积最小.
4. 求由曲线 $y=x^{\frac{3}{2}}$, 直线 $x=4$ 及 $x$ 轴所围图形绕 $y$ 轴旋转而成的旋转体的体积.
5. 求圆盘 $(x-2)^{2}+y^{2} \leqslant 1$ 绕 $y$ 轴旋转而成的旋转体的体积.
6. 求执物线 $y=\frac{1}{2} x^{2}$ 被圆 $x^{2}+y^{2}=3$ 所截下的有限部分的弧长.
7. 半径为 $r$ 的球沉人水中,球的上部与水面相切, 球的密度与水相同,现将球从水中取 出, 需作多少功?
8. 边长为 $a$ 和 $b$ 的矩形薄板, 与液面成 $\alpha$ 角斜沉于液体内, 长边平行于液面而位于深 $h$ 处、设 $a>b$, 液体的密度为 $\rho$, 试求薄板每面所受的压力.
9. 设星形线 $x=a \cos ^{3} t, y=a \sin ^{3} t$ 上每一点处的线密度的大小等于该点到原点距离的 立方,在原点 $O$ 处有一单位质点,求星形线在第一象限的弧段对这质点的引力.

## 第七章 微分方 程

函数是客观事物的内部联系在数量方面的反映, 利用函数关系又可以对客 观事物的规律性进行研究. 因此如何寻求函数关系, 在实践中具有重要意义. 在 许多问题中,往往不能直接找出所篮要的函数关系, 但是根据问题所提供的情 况,有时可以列出含有要找的函数及其导数的关系式.这样的关系式就是所谓微 分方程. 微分方程建立以后, 对它进行研究, 找出未知函数来, 这就是解微分方 程. 本章主要介绍微分方程的一些基本概念和几种常用的微分方程的解法.

## 第一节 微分方程的基本概念

下面我们通过几何、力学及物理学中的几个具体例题来说明微分方程的基 本概念.

例 1 一曲线通过点 $(1,2)$, 且在该曲线上任一点 $M(x, y)$ 处的切线的斜率 为 $2 x$, 求这曲线的方程.

解 设所求曲线的方程为 $y=\varphi(x)$. 根据导数的几何意义, 可知未知函数 $y=\varphi(x)$ 应满足关系式

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=2 x .
$$

此外,未知函数 $y=\varphi(x)$ 还应满足下列条件:

$$
x=1 \text { 时, } y=2 \text {. }
$$

把(1)式两端积分, 得

$$
y=\int 2 x \mathrm{~d} x \text { 即 } y=x^{2}+C,
$$

其中 $C$ 是任意常数.

把条件“ $x=1$ 时, $y=2$ ”代入(3)式,得

$$
2=1^{2}+C,
$$

由此定出 $C=1$. 把 $C=1$ 代入(3)式, 即得所求曲线方程

$$
y=x^{2}+1 \text {. }
$$

例 2 列车在平直线路上以 $20 \mathrm{~m} / \mathrm{s}$ (相当于 $72 \mathrm{~km} / \mathrm{h}$ ) 的速度行驶; 当制动 时列车获得加速度 $-0.4 \mathrm{~m} / \mathrm{s}^{2}$. 问开始制动后多少时间列车才能停住以及列车 在这段时间里行驶了多少路程?

解 设列车在开始制动后 $t \mathrm{~s}$ 时行驶了 $s \mathrm{~m}$. 根据题意, 反映制动阶段列车 运动规律的函数 $s=s(t)$ 应满足关系式

$$
\frac{\mathrm{d}^{2} s}{\mathrm{~d} t^{2}}=-0.4 \text {. }
$$

此外,未知函数 $s=s(t)$ 还应满足下列条件:

$$
t=0 \text { 时, } s=0, v=\frac{\mathrm{d} s}{\mathrm{~d} t}=20 \text {. }
$$

把(5)式两端积分一次, 得

$$
v=\frac{\mathrm{d} s}{\mathrm{~d} t}=-0.4 t+C_{1} ;
$$

再积分一次, 得

$$
s=-0.2 t^{2}+C_{1} t+C_{2},
$$

这里 $C_{1}, C_{2}$ 都是任意常数.

把条件“ $t=0$ 时, $v=20$ ”代入(7)式, 得

$$
20=C_{1} ;
$$

把条件“ $t=0$ 时, $s=0$ ”代入(8)式,得

$$
0=C_{2} \text {. }
$$

把 $C_{1}, C_{2}$ 的值代入(7)及(8)式,得

$$
\begin{gathered}
v=-0.4 t+20, \\
s=-0.2 t^{2}+20 t .
\end{gathered}
$$

在 (9)式中令 $v=0$, 得到列车从开始制动到完全停住所需的时间

$$
t=\frac{20}{0.4}=50(\mathrm{~s}) \text {. }
$$

再把 $t=50$ 代入 $(10)$ 式,得到列车在制动阶段行驶的路程

$$
s=-0.2 \times 50^{2}+20 \times 50=500(\mathrm{~m}) .
$$

上述两个例子中的关系式 (1) 和 (5) 都含有未知函数的导数,它们都是微分 方程.一般的,凡表示未知函数、未知函数的导数与自变量之间的关系的方程, 叫 做微分方程,有时也简称方程.

微分方程中所出现的未知函数的最高阶导数的阶数,叫做微分方程的阶.例 如, 方程 (1) 是一阶微分方程; 方程 (5) 是二阶微分方程. 又如, 方程

$$
x^{3} y^{\prime \prime \prime}+x^{2} y^{\prime \prime}-4 x y^{\prime}=3 x^{2}
$$

是三阶微分方程;方程

$$
y^{(4)}-4 y^{\prime \prime \prime}+10 y^{\prime \prime}-12 y^{\prime}+5 y=\sin 2 x
$$

是四阶微分方程.

一般的, $n$ 阶微分方程的形式是

$$
F\left(x, y, y^{\prime}, \cdots, y^{(n)}\right)=0 .
$$

这里必须指出, 在方程 (11) 中, $y^{(n)}$ 是必须出现的, 而 $x, y, y^{\prime}, \cdots, y^{(n-1)}$ 等变望 则可以不出现. 例如 $n$ 阶微分方程

$$
y^{(n)}+1=0
$$

中,除 $y^{(n)}$ 外,其他变量都没有出现.

如果能从方程 (11) 中解出最高阶导数, 则可得微分方程

$$
y^{(n)}=f\left(x, y, y^{\prime}, \cdots, y^{(n-1)}\right) \text {. }
$$

以后我们讨论的微分方程都是已解出最高阶导数的方程或能解出最高阶导数的 方程。

由前面的例子我们看到, 在研究某些实际问题时,首先要建立微分方程, 然 后找出满足微分方程的函数(解微分方程), 就是说, 找出这样的函数, 把这函数 代入微分方程能使该方程成为恒等式. 这个函数就叫做该微分方程的解. 确切地 说,设函数 $y=\varphi(x)$ 在区间 $I$ 上有 $n$ 阶连续导数,如果在区间 $I$ 上,

$$
F\left[x, \varphi(x), \varphi^{\prime}(x), \cdots, \varphi^{(n)}(x)\right] \equiv 0,
$$

那么函数 $y=\varphi(x)$ 就叫做微分方程 (11) 在区间 $I$ 上的解.

例如, 函数 (3) 和 (4) 都是微分方程 (1) 的解; 函数 (8) 和 (10) 都是微分方程 (5)的解.

如果微分方程的解中含有任意常数,且任意常数的个数与微分方程的阶数 相同 ${ }^{\circledR}$, 这样的解叫做微分方程的通解. 例如, 函数 (3) 是方程 (1) 的解, 它含有一 个任意常数,而方程 (1) 是一阶的,所以函数 (3) 是方程 (1) 的通解. 又如, 函数 (8) 是方程 (5) 的解, 它含有两个任意常数, 而方程 (5) 是二阶的, 所以函数 (8) 是方程 (5)的通解.

由于通解中含有任意常数,所以它还不能完全确定地反咉某一客观事物的 规律性. 要完全确定地反映客观事物的规律性, 必须确定这些常数的值. 为此, 要 根据问题的实际情况，提出确定这些常数的条件.例如，例 1 中的条件 (2)及例 2 中的条件 (6)便是这样的条件.

设微分方程中的未知函数为 $y=\varphi(x)$, 如果微分方程是一阶的,通常用来 确定任意常数的条件是

$$
x=x_{0} \text { 时, } y=y_{0} \text {, }
$$

或写成

$$
\left.y\right|_{\mathrm{r}:=x_{0}}=y_{0},
$$

其中 $x_{0} 、 y_{0}$ 都是给定的值; 如果微分方程是二阶的,通常用来确定任意常数的 条件是

$$
x=x_{0} \text { 时, } y=y_{0}, y^{\prime}=y_{0}^{\prime} \text {, }
$$

或写成

$$
\left.y\right|_{x=x_{0}}=y_{0},\left.y^{\prime}\right|_{x=x_{0}}=y_{0}^{\prime},
$$

其中 $x_{0} 、 y_{0}$ 和 $y_{0}^{\prime}$ 都是给定的值. 上述这种条件叫做初始条件.

确定了通解中的任意常数以后, 就得到微分方程的特解. 例如 (4) 式是方程 (1)满足条件 (2) 的特解: (10)式是方程 (5) 满足条件 (6) 的特解.

求微分方程 $y^{\prime}=f(x, y)$ 满足初始条件 $\left.y\right|_{x=x_{0}}=y_{0}$ 的特解这样一个问题, 叫做一阶微分方程的初值问题, 记作

$$
\left\{\begin{array}{l}
y^{\prime}=f(x, y), \\
\left.y\right|_{x=x_{0}}=y_{0} .
\end{array}\right.
$$

微分方程的解的图形是一条曲线, 叫做微分方程的积分曲线. 初值问题 (13) 的几何意义, 就是求微分方程的通过点 $\left(x_{0}, y_{11}\right)$ 的那条积分曲线. 二阶微分方程 的初值问题

$$
\left\{\begin{array}{l}
y^{\prime \prime}=f\left(x, y, y^{\prime}\right), \\
\left.y\right|_{x=x_{0}}=y_{0},\left.y^{\prime}\right|_{x=x_{0}}=y_{0}^{\prime}
\end{array}\right.
$$

的几何意义, 是求微分方程的通过点 $\left(x_{01}, y_{0}\right)$ 且在该点处的切线斜率为 $y_{0}^{\prime}$ 的那 条积分曲线.

例 3 验证: 函数

$$
x=C_{1} \cos k t+C_{2} \sin k t
$$

是微分方程

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+k^{2} x=0
$$

的解.

解 求出所给函数 (14) 的导数

$$
\begin{aligned}
\frac{\mathrm{d} x}{\mathrm{~d} t} & =-k C_{1} \sin k t+k C_{2} \cos k t, \\
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}} & =-k^{2} C_{1} \cos k t-k^{2} C_{2} \sin k t \\
& =-k^{2}\left(C_{1} \cos k t+C_{2} \sin k t\right) .
\end{aligned}
$$

把 $\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}$ 及 $x$ 的表达式代入方程 (15)，得

$$
-k^{2}\left(C_{1} \cos k t+C_{2} \sin k t\right)+k^{2}\left(C_{1} \cos k t+C_{2} \sin k t\right) \equiv 0 .
$$

函数 (14) 及其导数代入方程 (15) 后成为一个恒等式, 因此函数 (14) 是微分 方程 (15) 的解.

例 4 已知函数 (14) 当 $k \neq 0$ 时是微分方程 (15) 的通解, 求满足初始条件

$$
\left.x\right|_{1,=0}=A,\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=0}=0
$$

的特解.

解 将条件 “ $t=0$ 时, $x=A$ ”代入(14)式得

$$
C_{1}=A \text {. }
$$

将条件“ $t=0$ 时, $\frac{\mathrm{d} x}{\mathrm{~d} t}=0$ ”代入(16)式, 得

$$
C_{2}=0 \text {. }
$$

把 $C_{1} 、 C_{2}$ 的值代入 (14) 式, 就得所求的特解为

$$
x=A \cos k t \text {. }
$$

## 习 题 7-1

1. 试说出下列各微分方程的阶数:
(1) $x\left(y^{\prime}\right)^{2}-2 y y^{\prime}+x=0$;
(2) $x^{2} y^{\prime \prime}-x y^{\prime}+y=0$;
(3) $x y^{\prime \prime \prime}+2 y^{\prime \prime}+x^{2} y=0$;
(4) $(7 x-6 y) \mathrm{d} x+(x+y) \mathrm{d} y=0$;
(5) $L \frac{\mathrm{d}^{2} Q}{\mathrm{~d} t^{2}}+R \frac{\mathrm{d} Q}{\mathrm{~d} t}+\frac{Q}{C}=0$;
(6) $\frac{\mathrm{d} \rho}{\mathrm{d} \theta}+\rho=\sin ^{2} \theta$.
2. 指出下列各题中的函数是否为所给微分方程的解:
(1) $x y^{\prime}=2 y, y=5 x^{2}$;
(2) $y^{\prime \prime}+y=0, y=3 \sin x-4 \cos x$;
(3) $y^{\prime \prime}-2 y^{\prime}+y=0, y=x^{2} \mathrm{e}^{x}$;
(4) $y^{\prime \prime}-\left(\lambda_{1}+\lambda_{2}\right) y^{\prime}+\lambda_{1} \lambda_{2} y=0, y=C_{1} \mathrm{e}^{\lambda_{1} x}+C_{2} \mathrm{e}^{\lambda_{2}{ }^{x}}$.
3. 在下列各题中，验证所给二元方程所确定的函数为所给微分方程的解:

(1) $(x-2 y) y^{\prime}=2 x-y, x^{2}-x y+y^{2}=C$;

(2) $(x y-x) y^{\prime \prime}+x y^{\prime 2}+y y^{\prime}-2 y^{\prime}=0, y=\ln (x y)$.

4. 在下列各题中, 确定函数关系式中所含的参数, 使函数满足所给的初始条件:

(1) $x^{2}-y^{2}=C,\left.y\right|_{r=0}=5$;

(2) $y=\left(C_{1}+C_{2} x\right) \mathrm{e}^{2 x},\left.y\right|_{x=0}=0,\left.y^{\prime}\right|_{1=0}=1$;

(3) $y=C_{1} \sin \left(x-C_{2}\right),\left.y\right|_{x=\pi}=1,\left.y^{\prime}\right|_{x=\pi}=0$.

5. 写出由下列条件确定的曲线所满足的微分方程:

(1) 曲线在点 $(x, y)$ 处的切线的斜率等于该点㮴坐标的平方;

(2) 曲线上点 $P(x, y)$ 处的法线与 $x$ 轴的交点为 $Q$, 且线段 $P Q$ 被 $y$ 轴平分.

6. 用微分方程表示一物理命题: 某种气体的压强 $p$ 对于温度 $T$ 的变化率与压强成正 比, 与温度的平方成反比.

## 第二节 可分离变量的微分方程

本节至第四节,我们讨论一阶微分方程

$$
y^{\prime}=f(x, y)
$$

的一些解法.

一阶微分方程有时也写成如下的对称形式:

$$
P(x, y) \mathrm{d} x+Q(x, y) \mathrm{d} y=0 .
$$

在方程 (2)中,变量 $x$ 与 $y$ 对称, 它既可看作是以 $x$ 为自变量、 $y$ 为因变量的方 程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=-\frac{P(x, y)}{Q(x, y)}
$$

(这时 $Q(x, y) \neq 0$ ), 也可看作是以 $y$ 为自变量 、 $x$ 为因变量的方程

$$
\frac{\mathrm{d} x}{\mathrm{~d} y}=-\frac{Q(x, y)}{P(x, y)}
$$

（这时 $P(x, y) \neq 0$ ).

在第一节的例 1 中, 我们遇到一阶微分方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=2 x,
$$

或

$$
\mathrm{d} y=2 x \mathrm{~d} x .
$$

把上式两端积分就得到这个方程的通解

$$
y=x^{2}+C .
$$

但是并不是所有的一阶微分方程都能这样求解.例如,对于一阶微分方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=2 x y^{2}
$$

就不能像上面那样用直接对两端积分的方法求出它的通解. 这是什么缘故呢? 原因是方程 (3) 的右端含有与 $x$ 存在函数关系的变量 $y$,积分

$$
\int 2 x y^{2} \mathrm{~d} x
$$

求不出来, 这是困难所在. 为了解决这个困难, 在方程 (3) 的两端同时乘以 $\frac{\mathrm{d} x}{y^{2}}$, 使 方程 $(3)$ 变为

$$
\frac{\mathrm{d} y}{y^{2}}=2 x \mathrm{~d} x,
$$

这样,变量 $x$ 与 $y$ 已分离在等式的两端, 然后两端积分得

$$
\begin{aligned}
& -\frac{1}{y}=x^{2}+C, \\
& y=-\frac{1}{x^{2}+C},
\end{aligned}
$$

其中 $C$ 是任意常数.

可以验证, 函数 (4) 确实满足一阶微分方程 (3), 且含有一个任意常数,所以 它是方程 (3) 的通解.

一般的,如果一个一阶微分方程能写成

$$
g(y) \mathrm{d} y=f(x) \mathrm{d} x
$$

的形式, 就是说, 能把微分方程写成一端只含 $y$ 的函数和 $\mathrm{d} y$, 另一端只含 $x$ 的 函数和 $\mathrm{d} x$, 那么原方程就称为可分离变量的微分方程.

假定方程 (5) 中的函数 $g(y)$ 和 $f(x)$ 是连续的. 设 $y=\varphi(x)$ 是方程 $(5)$ 的 解, 将它代入 (5)中得到恒等式

$$
g[\varphi(x)] \varphi^{\prime}(x) \mathrm{d} x=f(x) \mathrm{d} x .
$$

将上式两端积分, 并由 $y=\varphi(x)$ 引进变監 $y$, 得

$$
\int g(y) \mathrm{d} y=\int f(x) \mathrm{d} x .
$$

设 $G(y)$ 及 $F(x)$ 依次为 $g(y)$ 及 $f(x)$ 的原函数, 于是有

$$
G(y)=F(x)+C \text {. }
$$

因此,方程 (5)的解满足关系式 (6). 反之,如果 $y=\Phi(x)$ 是由关系式 (6) 所确定 的隐函数, 那么在 $g(y) \neq 0$ 的条件下, $y=\Phi(x)$ 也是方程 (5) 的解, 事实上, 由隐 函数的求导法可知, 当 $g(y) \neq 0$ 时,

$$
\Phi^{\prime}(x)=\frac{F^{\prime}(x)}{G^{\prime}(y)}=\frac{f(x)}{g(y)},
$$

这就表示函数 $y=\Phi(x)$ 满足方程 (5). 所以,如果已分离变望的方程 (5) 中, $g(y)$ 和 $f(x)$ 是连续的, 且 $g(y) \neq 0$, 那么 (5) 式两端积分后得到的关系式 (6), 就用隐式给出了方程 (5) 的解, (6) 式就叫做微分方程 (5) 的隐式解. 又由于关系 式 (6)中含有任意常数,因此 (6)式所确定的隐函数是方程 (5) 的通解, 所以 (6)式 叫做微分方程 (5) 的隐式通解 (当 $f(x) \neq 0$ 时, (6) 式所确定的隐函数 $x=\Psi(y)$ 也可认为是方程 (5) 的解).

例 1 . 求微分方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=2 x y
$$

的通解.

解 方程 (7)是可分离变量的,分离变量后得

两端积分

$$
\frac{\mathrm{d} y}{y}=2 x \mathrm{~d} x \text {, }
$$

得

$$
\int \frac{\mathrm{d} y}{y}=\int 2 x \mathrm{~d} x \text {, }
$$

从而

$$
\ln |y|=x^{2}+C_{1} \text {, }
$$

因 $\pm \mathrm{e}^{c_{1}}$ 是任意非零常数, 又 $y \equiv 0$ 也是方程 (7) 的解; 故得方程 (7) 的通解

$$
y=C \mathrm{e}^{x^{2}}
$$

例 2 放射性元素铀由于不断地有原子放射出微粒子而变成其他元素, 铀 的含量就不断减少, 这种现象叫做衰恋. 由原子物理学知道, 铀的衰变速度与当 时未衰变的铀原子的含量 $M$ 成正比. 已知 $\iota=0$ 时铀的含墨为 $M_{0}$, 求在衰变过 程中铀含量 $M(t)$ 随时间 $t$ 变化的规律.

解 铀的衰变速度就是 $M(t)$ 对时间 $t$ 的导数 $\frac{\mathrm{d} M}{\mathrm{~d} t}$. 由于铀的衰变速度与其 含量成正比,故得微分方程

$$
\frac{\mathrm{d} M}{\mathrm{~d} \iota}=-\lambda M,
$$

其中 $\lambda(\lambda>0)$ 是常数, 叫做衰恋系数, $\lambda$ 前置负号是由于当 $t$ 增加时 $M$ 单调减 少, 即 $\frac{\mathrm{d} M}{\mathrm{~d} t}<0$ 的缘故.

按题意,初始条件为

$$
\left.M\right|_{t=0}=M_{0} .
$$

方程 $(8)$ 是可分离变量的. 分离变量后得

$$
\frac{\mathrm{d} M}{M}=-\lambda \mathrm{d} t .
$$

两端积分

$$
\int \frac{\mathrm{d} M}{M}=\int(-\lambda) \mathrm{d} t,
$$

以 $\ln C$ 表示任意常数,考虑到 $M>0$, 得

即

$$
\begin{gathered}
\ln M=-\lambda t+\ln C, \\
M=C \mathrm{e}^{-\lambda t} .
\end{gathered}
$$

这就是方程 (8) 的通解. 以初始条件代入上式, 得

$$
M_{0}=C \mathrm{e}^{0}=C,
$$

所以

$$
M=M_{0} \mathrm{e}^{-\lambda t} \text {, }
$$

这就是所求铀的衰变规律. 由此可见,铀的含量随时间的增加而按指数规律衰减 (图 7-1).

例 3 设降落伞从跳伞塔下落后, 所受空气阻力与速度成正比, 并设降落伞 离开跳伞塔时 $(t=0)$ 速度为零,求降落伞下落速度与时间的函数关系.

解 设降落伞下落速度为 $v(t)$. 降落伞在空中下落时, 同时受到重力 $P$ 与 阻力 $R$ 的作用 (图 7-2). 重力大小为 $m g$,方向与 $v$ 一致; 阻力大小为 $k v(k$ 为 比例系数), 方向与 $v$ 相反, 从而降落伞所受外力为

$$
F=m g-k v \text {. }
$$

根据牛顿第二运动定律

$$
F=m a
$$

(其中 $a$ 为加速度), 得函数 $v(t)$ 应满足的方程为

$$
m \frac{\mathrm{d} v}{\mathrm{~d} t}=m g-k v \text {. }
$$

按题意,初始条件为

$$
\left.v\right|_{t=0}=0 .
$$

方程 (9)是可分离变量的. 分离变量后得

两端积分

$$
\frac{\mathrm{d} v}{m g-k v}=\frac{\mathrm{d} t}{m},
$$

$$
\int \frac{\mathrm{d} v}{m g-k v}=\int \frac{\mathrm{d} t}{m} \text {, }
$$

考虑到 $m g-k v>0$, 得

$$
-\frac{1}{k} \ln (m g-k v)=\frac{t}{m}+C_{1},
$$

即

$$
m g-k v=\mathrm{e}^{-\frac{k}{m} t \cdot k c_{1}},
$$

或

$$
v=\frac{m g}{k}+C \mathrm{e}^{-\frac{k}{m} t} \quad\left(C=-\frac{\mathrm{e}^{k{ }^{k}}}{k}\right),
$$

这就是方程 (9) 的通解.

将初始条件 $\left.v\right|_{:=0}=0$ 代入 $(10)$ 式,得

$$
C=-\frac{m g}{k} \text {. }
$$

于是所求的特解为

$$
v=\frac{m g}{k}\left(1-\mathrm{e}^{-\frac{k}{m^{t}}}\right) \text {. }
$$

由(11) 可以看出, 随着时间 $t$ 的增大, 速度 $v$ 逐渐接近于常数 $\frac{m g}{k}$, 且不会超 过 $\frac{m g}{k}$, 也就是说, 跳苹后开始阶段是加速运动, 但以后逐渐接近于等速运动.

例 4 有高为 $1 \mathrm{~m}$ 的半球形容器, 水从它的底部小孔流出, 小孔横截面面积 为 $1 \mathrm{~cm}^{2}$ (图 7-3). 开始时容器内盛满了水, 求水从小孔流出过程中容器里水面 的高度 $h$ (水面与孔口中心间的距离) 随时间 $t$ 变化的规律, 并求水流完所需的 时间.

解 由水力学知道, 水从孔口流出的流墨 (即通过孔口横截面的水的体积 $V$ 对时间 $t$ 的变化率) $Q$ 可用下列公式计算：

$$
Q=\frac{\mathrm{d} V}{\mathrm{~d} t}=k S \sqrt{2 g h},
$$

其中 $k$ 为流量系数, 由实验测得 $k=0.62, S$ 为孔口横截面面积, $g$ 为重力加速 度.

另一方面, 设在微小时间间隔 $[t, t+\mathrm{d} t]$ 内, 水面高度由 $h$ 降至 $h+\mathrm{d} h(\mathrm{~d} h$ $<0)$, 则又可得到

$$
\mathrm{d} V=-\pi r^{2} \mathrm{~d} h,
$$

其中 $r$ 是时刻 $t$ 的水面半径 (图 7-3), 右端置负号是由于 $\mathrm{d} h<0$ 而 $\mathrm{d} V>0$ 的 缘故. 又因

$$
r=\sqrt{1^{2}-(1-h)^{2}}=\sqrt{2 h-h^{2}},
$$

所以 (13)式变成

$$
\mathrm{d} V=-\pi\left(2 h-h^{2}\right) \mathrm{d} h .
$$

比较 (12)和 (14)两式,得

$$
k S \sqrt{2 g h} \mathrm{~d} t=-\pi\left(2 h-h^{2}\right) \mathrm{d} h,
$$

这就是未知函数 $h=h(t)$ 应满足的微分方程.

此外, 开始时容器内的水是满的, 所以未知函数 $h=h(t)$ 还应满足下列初 始条件:

$$
\left.h\right|_{t=0}=1 .
$$

方程 (15) 是可分离变量的. 分离变量后得

$$
\mathrm{d} t=-\frac{\pi}{k S \sqrt{2 g}}\left(2 h^{\frac{1}{2}}-h^{\frac{3}{2}}\right) \mathrm{d} h .
$$

两端积分，得

$$
t=-\frac{\pi}{k S \sqrt{2 g}}\left(\frac{4}{3} h^{\frac{3}{2}}-\frac{2}{5} h^{\frac{5}{2}}+C\right),
$$

其中 $C$ 是任意常数.

把初始条件(16)代入(17)式,得

$$
C=-\frac{4}{3}+\frac{2}{5}=-\frac{14}{15} \text {. }
$$

把所得的 $C$ 值代入(17)式并化简, 就得

$$
t=\frac{14 \pi}{15 k S \sqrt{2 g}}\left(1-\frac{10}{7} h^{\frac{3}{2}}+\frac{3}{7} h^{\frac{5}{2}}\right) .
$$

以 $k=0.62, S=10^{-4} \mathrm{~m}^{2}, g=9.8 \mathrm{~m} / \mathrm{s}^{2}$ 代入上式,计算后可得

$$
t=1.068 \times 10^{4}\left(1-\frac{10}{7} h^{\frac{3}{2}}+\frac{3}{7} h^{\frac{5}{2}}\right)(\mathrm{s}) .
$$

上式表达了水从小孔流出的过程中容器内水面高度 $h$ 与时间 $t$ 之间的函数 关系.由此可知水流完所需的时间为

$$
t=1.068 \times 10^{4} \mathrm{~s}=2 \mathrm{~h} 58 \mathrm{~min} .
$$

这里还要指出, 在例 4 中我们是通过对微小量 $\mathrm{d} V$ 的分析得到微分方程 (15) 的.这种微小星分析的方法,也是建立微分方程的一种常用方法.

## 习 题 7-2

1. 求下列微分方程的通解：
(1) $x y^{\prime}-y \ln y=0$;
(2) $3 x^{2}+5 x-5 y^{\prime}=0$;
(3) $\sqrt{1-x^{2}} y^{\prime}=\sqrt{1-y^{2}}$;
(4) $y^{\prime}-x y^{\prime}=a\left(y^{2}+y^{\prime}\right)$;
(5) $\sec ^{2} x \tan y \mathrm{~d} x+\sec ^{2} y \tan x \mathrm{~d} y=0$;
(6) $\frac{\mathrm{d} y}{\mathrm{~d} x}=10^{x+y}$;
(7) $\left(\mathrm{e}^{x^{+x}}-\mathrm{e}^{x}\right) \mathrm{d} x+\left(\mathrm{e}^{x+x}+\mathrm{e}^{x}\right) \mathrm{d} y=0$;
(8) $\cos x \sin y \mathrm{~d} x+\sin x \cos y \mathrm{~d} y=0$;
(9) $(y+1)^{2} \frac{\mathrm{d} y}{\mathrm{~d} x}+x^{3}=0$;
(10) $y \mathrm{~d} x+\left(x^{2}-4 x\right) \mathrm{d} y=0$.
2. 求下列微分方程满足所给初始条件的特解:

(1) $y^{\prime}=\mathrm{e}^{2 x-y},\left.y\right|_{x=0}=0$;

(2) $\cos x \sin y \mathrm{~d} y=\cos y \sin x \mathrm{~d} x,\left.y\right|_{r=11}=\frac{\pi}{4}$;

(3) $y^{\prime} \sin x=\left.y \ln y \cdot y\right|_{,=\frac{\pi}{2}}=\mathrm{e}$;

(4) $\cos y \mathrm{~d} x+\left(1+\mathrm{e}^{-x}\right) \sin y \mathrm{~d} y=0,\left.y\right|_{,=01}=\frac{\pi}{4}$;

(5) $x \mathrm{~d} y+2 y \mathrm{~d} x=0,\left.y\right|_{,>2}=1$.

3. 有一盛满了水的圆锥形漏斗, 高为 $10 \mathrm{~cm}$, 顶角为 $60^{\circ}$, 漏斗下面有面积为 $0.5 \mathrm{~cm}^{2}$ 的 孔，求水面高度变化的规律及水流完所需的时间.
4. 质些为 $1 \mathrm{~g}$ 的质点受外力作用作直线运动. 这外力和时间成正比, 和质点运动的速度 成反比. 在 $t=10 \mathrm{~s}$ 时. 速度等于 $50 \mathrm{~cm} / \mathrm{s}$, 外力为 $4 \mathrm{~g} \cdot \mathrm{cm} / \mathrm{s}^{2}$, 问从运动开始经过了一分钟后的 速度是多少?

经过 1600 年后, 只余原始证 $R_{\mathrm{u}}$ 的一半. 试求锚的现存些 $R$ 与时间 $t$ 的函数关系.

6. 一曲线通过点 $(2,3)$, 它在两坐标秞间的任一切线线段均被切点所平分, 求这曲线方程.
7. 小船从河边点 $O$ 处出发驶向对岸 (两岸为平行直线). 设船速为 $a$, 船行方向始终与河 岸垂直, 又设河宽为 $h$, 河中任一点处的水流速度与该点到两岸距离的乘积成正比(比例系数 为 $k$ ). 求小船的航行路线.

## 第三节 齐次 方 程

## 一、齐次方程

如果一阶微分方程可化成

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\varphi\left(\frac{y}{x}\right)
$$

的形式,那么就称这方程为齐次方程, 例如

$$
\left(x y-y^{2}\right) \mathrm{d} x-\left(x^{2}-2 x y\right) \mathrm{d} y=0
$$

是齐次方程,因为它可化成

即

$$
\begin{gathered}
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{x y-y^{2}}{x^{2}-2 x y}, \\
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\frac{y}{x}-\left(\frac{y}{x}\right)^{2}}{1-2\left(\frac{y}{x}\right)} .
\end{gathered}
$$

在齐次方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\varphi\left(\frac{y}{x}\right)
$$

中,引进新的未知函数

$$
u=\frac{y}{x},
$$

就可把它化为可分离变量的方程. 因为由 (2) 有

$$
y=u x, \frac{\mathrm{d} y}{\mathrm{~d} x}=u+x \frac{\mathrm{d} u}{\mathrm{~d} x},
$$

代入方程 (1),便得方程

即

$$
\begin{aligned}
& u+x \frac{\mathrm{d} u}{\mathrm{~d} x}=\varphi(u), \\
& x \frac{\mathrm{d} u}{\mathrm{~d} x}=\varphi(u)-u .
\end{aligned}
$$

分离变量,得

$$
\frac{\mathrm{d} u}{\varphi(u)-u}=\frac{\mathrm{d} x}{x} .
$$

两端积分, 得

$$
\int \frac{\mathrm{d} u}{\varphi(u)-u}=\int \frac{\mathrm{d} x}{x} .
$$

求出积分后, 再以 $\frac{y}{x}$ 代替 $u$, 便得所给齐次方程的通解.

例 1 解方程

$$
y^{2}+x^{2} \frac{\mathrm{d} y}{\mathrm{~d} x}=x y \frac{\mathrm{d} y}{\mathrm{~d} x} .
$$

解 原方程可写成

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{y^{2}}{x y-x^{2}}=\frac{\left(\frac{y}{x}\right)^{2}}{\frac{y}{x}-1},
$$

因此是齐次方程. 令 $\frac{y}{x}=u$, 则

$$
y=u x, \frac{\mathrm{d} y}{\mathrm{~d} x}=u+x \frac{\mathrm{d} u}{\mathrm{~d} x},
$$

于是原方程变为

即

$$
\begin{gathered}
u+x \frac{\mathrm{d} u}{\mathrm{~d} x}=\frac{u^{2}}{u-1} \\
x \frac{\mathrm{d} u}{\mathrm{~d} x}=\frac{u}{u-1} .
\end{gathered}
$$

分离变量, 得

两端积分, 得 或写为

$$
\begin{gathered}
\left(1-\frac{1}{u}\right) \mathrm{d} u=\frac{\mathrm{d} x}{x} . \\
u-\ln |u|+C=\ln |x|, \\
\ln |x u|=u+C .
\end{gathered}
$$

以 $\frac{y}{x}$ 代上式中的 $u$, 便得所给方程的通解为

$$
\ln |y|=\frac{y}{x}+C .
$$

例 2 探照灯的聚光镜的镜面是一张旋转曲面, 它的形状由 $x O y$ 坐标面上 的一条曲线 $L$ 绕 $x$ 轴旋转而成. 按聚光镜性能的要求, 在其旋转轴 ( $x$ 轴) 上一 点 $O$ 处发出的一切光线,经它反射后都与旋转轴平行. 求曲线 $L$ 的方程.

解 将光源所在之 $O$ 点取作坐标原点 (如图 7-4), 且曲线 $L$ 位于 $y \geqslant 0$ 范 围内.

设点 $M(x, y)$ 为 $L$ 上的任一点, 点 $O$ 发出的某条光线经点 $M$ 反射后是一 条与 $x$ 轴平行的直线 $M S$. 又设过点 $M$ 的切线 $A T$ 与 $x$ 轴的夹角为 $\alpha$. 根据题 意, $\angle S M T=\alpha$. 另一方面, $\angle O M A$ 是人射角的余角, $\angle S M T$ 是反射角的余角, 于是由光学中的反射定律有 $\angle O M A=\angle S M T=\alpha$. 从而 $A O=O M$, 但 $A O=$ $A P-O P=P M \cot \alpha-O P=\frac{y}{y^{\prime}}-x$, 而 $O M=\sqrt{x^{2}+y^{2}}$.于是得微分方程

$$
\frac{y}{y^{\prime}}-x=\sqrt{x^{2}+y^{2}} .
$$

把 $x$ 看作因变量, $y$ 看作自变量,当 $y>0$ 时, 上式即为

$$
\frac{\mathrm{d} x}{\mathrm{~d} y}=\frac{x}{y}+\sqrt{\left(\frac{x}{y}\right)^{2}+1},
$$

这是齐次方程. 令 $\frac{x}{y}=v$, 则 $x=y v, \frac{\mathrm{d} x}{\mathrm{~d} y}=v+y \frac{\mathrm{d} v}{\mathrm{~d} y}$, 代 人上式,得

$$
v+y \frac{\mathrm{d} v}{\mathrm{~d} y}=v+\sqrt{v^{2}+1}
$$

即

$$
y \frac{\mathrm{d} v}{\mathrm{~d} y}=\sqrt{v^{2}+1} .
$$

分离变量, 得

$$
\frac{\mathrm{d} v}{\sqrt{v^{2}+1}}=\frac{\mathrm{d} y}{y} \text {. }
$$

积分, 得

$$
\ln \left(v+\sqrt{v^{2}+1}\right)=\ln y-\ln C,
$$

或

$$
v+\sqrt{v^{2}+1}=\frac{y}{C} \text {. }
$$

由

$$
\left(\frac{y}{C}-v\right)^{2}=v^{2}+1 \text {, }
$$

得

$$
\frac{y^{2}}{C^{2}}-\frac{2 y v}{C}=1 \text {, }
$$

以 $y v=x$ 代入上式,得

$$
y^{2}=2 C\left(x+\frac{C}{2}\right) .
$$

这是以 $x$ 轴为轴、焦点在原点的抛物线.

## " 二、可化为齐次的方程

方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{a x+b y+c}{a_{1} x+b_{1} y+c_{1}}
$$

当 $c=c_{1}=0$ 时是齐次的, 否则不是齐次的: 在非齐次的情形, 可用下列变换把它 化为齐次方程: 令

$$
x=X+h, y=Y+k,
$$

其中 $h$ 及 $k$ 是待定的常数. 于是 从而方程 (3)成为

$$
\mathrm{d} x=\mathrm{d} X, \mathrm{~d} y=\mathrm{d} Y,
$$

$$
\frac{\mathrm{d} Y}{\mathrm{~d} X}=\frac{a X+b Y+a h+b k+c}{a_{1} X+b_{1} Y+a_{1} h+b_{1} k+c_{1}}
$$

如果方程组

$$
\left\{\begin{array}{l}
a h+b k+c=0 \\
a_{1} h+b_{1} k+c_{1}=0
\end{array}\right.
$$

的系数行列式 $\left|\begin{array}{ll}a & b \\ a_{1} & b_{1}\end{array}\right| \neq 0$, 即 $\frac{a_{1}}{a} \neq \frac{b_{1}}{b}$, 那么可以定出 $h$ 及 $k$ 使它们满足上述 方程组.这样,方程 (3)便化为齐次方程

$$
\frac{\mathrm{d} Y}{\mathrm{~d} X}=\frac{a X+b Y}{a_{1} X+b_{1} Y} .
$$

求出这齐次方程的通解后, 在通解中以 $x-h$ 代 $X, y-k$ 代 $Y$, 便得方程 (3)的通解.

当 $\frac{a_{1}}{a}=\frac{b_{1}}{b}$ 时, $h$ 及 $k$ 无法求得,因此上述方法不能应用. 但这时令 $\frac{a_{1}}{a}=\frac{b_{1}}{b}=\lambda$, 从而方程(3)可写成

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{a x+b y+c}{\lambda(a x+b y)+c_{1}} \text {. }
$$

引人新变量 $v=a x+b y$,则

$$
\frac{\mathrm{d} v}{\mathrm{~d} x}=a+b \frac{\mathrm{d} y}{\mathrm{~d} x} \text { 或 } \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{b}\left(\frac{\mathrm{d} v}{\mathrm{~d} x}-a\right) .
$$

于是方程 (3)成为

$$
\frac{1}{b}\left(\frac{\mathrm{d} v}{\mathrm{~d} x}-a\right)=\frac{v+c}{\lambda v+c_{1}}
$$

这是可分离变量的方程.

以上所介绍的方法可以应用于更一般的方程

例 3 解方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=f\left(\frac{a x+b y+c}{a_{1} x+b_{1} y+c_{1}}\right) .
$$

$$
(2 x+y-4) \mathrm{d} x+(x+y-1) \mathrm{d} y=0 .
$$

解 所给方程属方程 (3) 的类型. 令 $x=X+h, y=Y+k$, 则 $\mathrm{d} x=\mathrm{d} X$, $\mathrm{d} y=\mathrm{d} Y$,代入原方程得

$$
(2 X+Y+2 h+k-4) \mathrm{d} X+(X+Y+h+k-1) \mathrm{d} Y=0 .
$$

解方程组

$$
\left\{\begin{array}{l}
2 h+k-4=0 \\
h+k-1=0
\end{array}\right.
$$

得 $h=3, k=-2$. 令 $x=X+3, y=Y-2$, 原方程成为

$$
(2 X+Y) \mathrm{d} X+(X+Y) \mathrm{d} Y=0 \text {, }
$$

或

$$
\frac{\mathrm{d} Y}{\mathrm{~d} X}=-\frac{2 X+Y}{X+Y}=-\frac{2+\frac{Y}{X}}{1+\frac{Y}{X}},
$$

这是齐次方程.

令 $\frac{Y}{X}=u$, 则 $Y=u X, \frac{\mathrm{d} Y}{\mathrm{~d} X}=u+X \frac{\mathrm{d} u}{\mathrm{~d} X}$, 于是方程变为

或

$$
\begin{gathered}
u+X \frac{\mathrm{d} u}{\mathrm{~d} X}=-\frac{2+u}{1+u}, \\
X \frac{\mathrm{d} u}{\mathrm{~d} X}=-\frac{2+2 u+u^{2}}{1+u} \\
-\frac{u+1}{u^{2}+2 u+2} \mathrm{~d} u=\frac{\mathrm{d} X}{X} .
\end{gathered}
$$

分离变量得

积分得

$$
\ln C_{1}-\frac{1}{2} \ln \left(u^{2}+2 u+2\right)=\ln |X| \text {, }
$$

于是

$$
\frac{C_{1}}{\sqrt{u^{2}+2 u+2}}=|X| \text {, }
$$

或

$$
C_{2}=X^{2}\left(u^{2}+2 u+2\right) \quad\left(C_{2}=C_{1}^{2}\right) \text {, }
$$

即

$$
Y^{2}+2 X Y+2 X^{2}=C_{2} \text {. }
$$

以 $X=x-3, Y=y+2$ 代入上式并化简, 得

$$
2 x^{2}+2 x y+y^{2}-8 x-2 y=C \quad\left(C=C_{2}-10\right) .
$$

## 习 题 7-3

1. 求下列齐次方程的通解:
(1) $x y^{\prime}-y-\sqrt{y^{2}-x^{2}}=0$;
(2) $x \frac{\mathrm{d} y}{\mathrm{~d} x}=y \ln \frac{y}{x}$;
(3) $\left(x^{2}+y^{2}\right) \mathrm{d} x-x y \mathrm{~d} y=0$;
(4) $\left(x^{3}+y^{3}\right) \mathrm{d} x-3 x y^{2} \mathrm{~d} y=0$;
(5) $\left(2 x \sin \frac{y}{x}+3 y \cos \frac{y}{x}\right) \mathrm{d} x-3 x \cos \frac{y}{x} \mathrm{~d} y=0$;
(6) $\left(1+2 \mathrm{e}^{\frac{x}{y}}\right) \mathrm{d} x+2 \mathrm{e}^{\frac{x}{x}}\left(1-\frac{x}{y}\right) \mathrm{d} y=0$.
2. 求下列齐饮方程满足所给初始条件的特解:
(1) $\left(y^{2}-3 x^{2}\right) \mathrm{d} y+2 x y \mathrm{~d} . x=0,\left.y\right|_{.=0}=1$ ；
(2) $y^{\prime}=\frac{x}{y}+\frac{y}{x},\left.y\right|_{r=1}=2$;
(3) $\left(x^{2}+2 x y-y^{2}\right) \mathrm{d} x+\left(y^{2}+2 x y-x^{2}\right) \mathrm{d} y=0,\left.y\right|_{, r=1}=1$. 3. 设有联结点 $O(0,0)$ 和 $A(1,1)$ 的一段向上凸的曲线弧 $\overparen{O A}$, 对于 $\overparen{O A}$ 上任一点 $P(x, y)$, 曲线弧 $\overparen{O P}$ 与直线段 $\overline{O P}$ 所围图形的面积为 $x^{2}$, 求曲线弧 $\overparen{O A}$ 的方程.

4. 化下列方程为齐次方程,并求出通解:

(1) $(2 x-5 y+3) \mathrm{d} x-(2 x+4 y-6) \mathrm{d} y=0$;

(2) $(x-y-1) \mathrm{d} x+(4 y+x-1) \mathrm{d} y=0$;

(3) $(3 y-7 x+7) \mathrm{d} x+(7 y-3 x+3) \mathrm{d} y=0$;

(4) $(x+y) \mathrm{d} x+(3 x+3 y-4) \mathrm{d} y=0$.

## 第四节 一阶线性微分方程

## 一、线性方程

方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}+P(x) y=Q(x)
$$

叫做一阶线性微分方程, 因为它对于未知函数 $y$ 及其导数是一次方程. 如果 $Q(x) \equiv 0$, 则方程 (1) 称为齐次的; 如果 $Q(x) \neq 0$, 则方程 (1) 称为非齐次的.

设 (1) 为非齐次线性方程. 为了求出非齐次线性方程 (1) 的解, 我们先把 $Q(x)$ 换成零而写出方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}+P(x) y=0 .
$$

方程 (2) 叫做对应于非齐次线性方程 (1) 的齐次线性方程. 方程 (2) 是可分离变量 的, 分离变量后得

$$
\frac{\mathrm{d} y}{y}=-P(x) \mathrm{d} x
$$

两端积分，得

$$
\ln |y|=-\int P(x) \mathrm{d} x+C_{1} \text {, }
$$

或 $y=C \mathrm{e}^{-\int P(x) d x}\left(C= \pm \mathrm{e}^{C_{1}}\right)$,

这是对应的齐次线性方程 (2) 的通解(1).

现在我们使用所谓常数变易法来求非齐次线性方程 (1) 的通解. 这方法是把 (2) 的通解中的 $C$ 换成 $x$ 的未知函数 $u(x)$, 即作变换

$$
y=u \mathrm{e}^{-\int P(\cdot x) \mathrm{d} x} \text {, }
$$

(1) 这里记号 $\int P(x) \mathrm{d} x$ 表示 $P(x)$ 的某个确定的原函数.

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=u^{\prime} \mathrm{e}^{-\int P(x) \mathrm{d} \cdot r}-u P(x) \mathrm{e}^{-\int P(x) \mathrm{d} \cdot r} .
$$

将 (3)和 (4)代入方程 (1)得

$$
u^{\prime} \mathrm{e}^{-\int P(x) d x}-u P(x) \mathrm{e}^{-\int P(x) \mathrm{d} x}+P(x) u \mathrm{e}^{-\int P(x) d x}=Q(x),
$$

即

$$
u^{\prime} \mathrm{e}^{-\int P(x) \mathrm{d} \cdot r}=Q(x), u^{\prime}=Q(x) \mathrm{e}^{\int P(x) \mathrm{d} x} .
$$

两端积分, 得

$$
u=\int Q(x) \mathrm{e}^{\int P(x) \mathrm{d} x} \mathrm{~d} x+C .
$$

把上式代入 (3),便得非齐次线性方程 (1)的通解

$$
y=\mathrm{e}^{-\int P(x) \mathrm{d} x}\left(\int Q(x) \mathrm{e}^{\int P(. x) \mathrm{d} \cdot x} \mathrm{~d} x+C\right) .
$$

将(5)式改写成两项之和

$$
y=C \mathrm{e}^{-\int P(x) \mathrm{d} x}+\mathrm{e}^{-\int P(\cdot x) \mathrm{d} \cdot \mathrm{\int}} \int Q(x) \mathrm{e}^{\int P(x) \mathrm{d} \cdot x} \mathrm{~d} x,
$$

上式右端第一项是对应的齐次线性方程 (2)的通解,第二项是非齐次线性方程 (1)的一个特解 (在 (1) 的通解 (5) 中取 $C=0$ 便得到这个特解). 由此可知，一阶 非齐次线性方程的通解等于对应的齐次方程的通解与非齐次方程的一个特解之 和.

例 1 求方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}-\frac{2 y}{x+1}=(x+1)^{\frac{5}{2}}
$$

的通解.

解 这是一个非齐次线性方程. 先求对应的齐次方程的通解.

$$
\begin{gathered}
\frac{\mathrm{d} y}{\mathrm{~d} x}-\frac{2}{x+1} y=0, \\
\frac{\mathrm{d} y}{y}=\frac{2 \mathrm{~d} x}{x+1}, \\
\ln y=2 \ln (x+1)+\ln C, \\
y=C(x+1)^{2} .
\end{gathered}
$$

用常数变易法,把 $C$ 换成 $u$,即令

$$
y=u(x+1)^{2},
$$

那么

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=u^{\prime}(x+1)^{2}+2 u(x+1),
$$

代入所给非齐次方程,得

两端积分, 得

$$
\begin{gathered}
u^{\prime}=(x+1)^{\frac{1}{2}} . \\
u=\frac{2}{3}(x+1)^{\frac{3}{2}}+C .
\end{gathered}
$$

再把上式代入 (6)式, 即得所求方程的通解为

$$
y=(x+1)^{2}\left[\frac{2}{3}(x+1)^{\frac{3}{2}}+C\right] .
$$

例 2 有一个电路如图 7-5 所示, 其中电源电动势为 $E=E_{n 1} \sin \omega t$ $\left(E_{\mathrm{m}} 、 \omega\right.$ 都是常量), 电阻 $R$ 和电感 $L$ 都是常量. 求电流 $i(t)$.

解 (i) 列方程 由电学知道, 当电流变化时, $L$ 上有感应电动势 $-L \frac{\mathrm{d} i}{\mathrm{~d} t}$. 由回路电压定律得出

$$
E-L \frac{\mathrm{d} i}{\mathrm{~d} t}-i R=0,
$$

即

$$
\frac{\mathrm{d} i}{\mathrm{~d} t}+\frac{R}{L} i=\frac{E}{L} .
$$

把 $E=E_{n} \sin \omega t$ 代入上式，得

$$
\frac{\mathrm{d} i}{\mathrm{~d} t}+\frac{R}{L} i=\frac{E_{\mathrm{m}}}{L} \sin \omega t .
$$

未知函数 $i(t)$ 应满足方程 (7). 此外, 设开关 $\mathrm{S}$ 闭合的时刻为 $t=0$, 这时 $i(t)$ 还应该满足初始条件

$$
\left.i\right|_{t=0}=0 \text {. }
$$

(ii) 解方程 方程 (7)是一个非齐次线性方程. 可以先求出对应的齐次方程 的通解, 然后用常数变易法求非齐次方程的通解. 但是, 也可以直接应用通解公 式(5) 来求解. 这里 $P(t)=\frac{R}{L}, Q(t)=\frac{E_{\mathrm{n}}}{L} \sin \omega t$, 代入公式 $(5)$, 得

$$
i(t)=\mathrm{e}^{-\frac{R}{L^{t}}}\left(\int \frac{E_{\mathrm{m}}}{L} \mathrm{e}^{\frac{R}{L^{\prime}}} \sin \omega t \mathrm{~d} t+C\right) .
$$

应用分部积分法, 得

$$
\int \mathrm{e}^{\frac{R}{L^{t}}} \sin \omega t \mathrm{~d} t=\frac{\mathrm{e}^{\frac{R}{L^{\prime}}}}{R^{2}+\omega^{2} L^{2}}\left(R L \sin \omega t-\omega L^{2} \cos \omega t\right),
$$

将上式代入前式并化简,得方程 (7)的通解

$$
i(t)=\frac{E_{\mathrm{m}}}{R^{2}+\omega^{2} L^{2}}(R \sin \omega t-\omega L \cos \omega t)+C \mathrm{e}^{-\frac{R}{L T}},
$$

其中 $C$ 为任意常数.

将初始条件 (8)代入上式, 得

$$
C=\frac{\omega L E_{\mathrm{nt}}}{R^{2}+\omega^{2} L^{2}},
$$

因此,所求函数 $i(t)$ 为

$$
i(t)=\frac{\omega L E_{\mathrm{m}}}{R^{2}+\omega^{2} L^{2}} \mathrm{e}^{-\frac{R}{L^{t}}}+\frac{E_{\mathrm{m}}}{R^{2}+\omega^{2} L^{2}}(R \sin \omega t-\omega L \cos \omega t) .
$$

为了便于说明(9)式所反映的物理现象,下面把 $i(\iota)$ 中第二项的形式稍加 改变.

$$
\text { 令 } \cos \varphi=\frac{R}{\sqrt{R^{2}+\omega^{2} L^{2}}}, \sin \varphi=\frac{\omega L}{\sqrt{R^{2}+\omega^{2} L^{2}}},
$$

于是(9)式可写成

$$
i(t)=\frac{\omega L E_{\mathrm{m}}}{R^{2}+\omega^{2} L^{2}} \mathrm{e}^{-\frac{R}{L^{t}}}+\frac{E_{\mathrm{m}}}{\sqrt{R^{2}+\omega^{2} L^{2}}} \sin (\omega t-\varphi),
$$

其中.

$$
\varphi=\arctan \frac{\omega L}{R} \text {. }
$$

当 $t$ 增大时, 上式右端第一项 (叫做暂态电流) 逐渐衰减而趋于零; 第二项 (叫做稳态电流) 是正弦函数, 它的周期和电动势的周期相同、而相角落后 $\varphi$.

在上节中, 对于齐次方程 $y^{\prime}=f\left(\frac{y}{x}\right)$, 我们通过变量代换 $y=x u$, 把它化为 变量可分离的方程, 然后分离变量, 经积分求得通解. 在本节中, 对于一阶非齐次 线性方程

$$
y^{\prime}+P(x) y=Q(x),
$$

我们通过解对应的齐次线性方程找到变量代换

$$
y=u \mathrm{e}^{-\int p(. r) \mathrm{d} \cdot r},
$$

利用这一代换,把非齐次线性方程化为变量可分离的方程,然后经积分求得通 解.

利用变量代换(因变量的变量代换或自变量的变量代换),把一个微分方程 化为变舅可分离的方程, 或化为已经知其求解步骤的方程, 这是解微分方程最常 用的方法.下面再举一个例子.

例 3 解方程 $\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{x+y}$.

解 若把所给方程变形为

$$
\frac{\mathrm{d} x}{\mathrm{~d} y}=x+y,
$$

即为一阶线性方程, 则按一阶线性方程的解法可求得通解.

也可用变量代换来解所给方程：

$$
\begin{aligned}
& \text { 令 } x+y=u \text {, 则 } y=u-x, \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} u}{\mathrm{~d} x}-1 \text {. 代入原方程, 得 } \\
& \frac{\mathrm{d} u}{\mathrm{~d} x}-1=\frac{1}{u}, \frac{\mathrm{d} u}{\mathrm{~d} x}=\frac{u+1}{u} \text {. }
\end{aligned}
$$

分离变量得

$$
\frac{u}{u+1} \mathrm{~d} u=\mathrm{d} x
$$

两端积分得

$$
u-\ln |u+1|=x+C .
$$

以 $u=x+y$ 代入上式, 即得

$$
\begin{gathered}
y-\ln |x+y+1|=C, \\
\text { 或 } \quad x=C_{1} \mathrm{e}^{y}-y-1 \quad\left(C_{1}= \pm \mathrm{e}^{-c}\right) .
\end{gathered}
$$

## * 二、伯努利方程

方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}+P(x) y=Q(x) y^{n} \quad(n \neq 0,1)
$$

叫做伯努利 (Bernoulli)方程. 当 $n=0$ 或 $n=1$ 时, 这是线性微分方程. 当 $n \neq 0$, $n \neq 1$ 时, 这方程不是线性的, 但是通过变量的代换, 便可把它化为线性的. 事实 上, 以 $y^{n}$ 除方程 (10)的两端, 得

$$
y^{-n} \frac{\mathrm{d} y}{\mathrm{~d} x}+P(x) y^{1-n}=Q(x) .
$$

容易看出, 上式左端第一项与 $\frac{\mathrm{d}}{\mathrm{d} x}\left(y^{1-n}\right)$ 只差一个常数因子 $1-n$, 因此我们引人 新的因变星

$$
z=y^{1-n},
$$

那么

$$
\frac{\mathrm{d} z}{\mathrm{~d} x}=(1-n) y^{-n} \frac{\mathrm{d} y}{\mathrm{~d} x} .
$$

用 $(1-n)$ 乘方程 $(11)$ 的两端,再通过上述代换便得线性方程

$$
\frac{\mathrm{d} z}{\mathrm{~d} x}+(1-n) P(x) z=(1-n) Q(x) .
$$

求出这方程的通解后, 以 $y^{1-n}$ 代 $z$ 便得到伯努利方程的通解.

例 4 求方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{y}{x}=a(\ln x) y^{2}
$$

的通解.

解 以 $y^{2}$ 除方程的两端, 得

即

$$
\begin{gathered}
y^{-2} \frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{1}{x} y^{-1}=a \ln x, \\
-\frac{\mathrm{d}\left(y^{-1}\right)}{\mathrm{d} x}+\frac{1}{x} y^{-1}=a \ln x,
\end{gathered}
$$

令 $z=y^{-1}$, 则上述方程成为

$$
\frac{\mathrm{d} z}{\mathrm{~d} x}-\frac{1}{x} z=-a \ln x .
$$

这是一个线性方程,它的通解为

$$
z=x\left[C-\frac{a}{2}(\ln x)^{2}\right] .
$$

以 $y^{-1}$ 代 $z$, 得所求方程的通解为

$$
y x\left[C-\frac{a}{2}(\ln x)^{2}\right]=1 .
$$

## 习 题 7-4

1. 求下列微分方程的通解:
(1) $\frac{\mathrm{d} y}{\mathrm{~d} x}+y=\mathrm{e}^{-x}$;
(2) $x y^{\prime}+y=x^{2}+3 x+2$;
(3) $y^{\prime}+y \cos x=\mathrm{e}^{-\sin x}$;
(4) $y^{\prime}+y \tan x=\sin 2 x$;
(5) $\left(x^{2}-1\right) y^{\prime}+2 x y-\cos x=0$;
(6) $\frac{d \rho}{d \theta}+3 \rho=2$;
(7) $\frac{\mathrm{d} y}{\mathrm{~d} x}+2 x y=4 x$;
(8) $y \ln y \mathrm{~d} x+(x-\ln y) \mathrm{d} y=0$;
(9) $(x-2) \frac{\mathrm{d} y}{\mathrm{~d} x}=y+2(x-2)^{3}$;
(10) $\left(y^{2}-6 x\right) \frac{\mathrm{d} y}{\mathrm{~d} x}+2 y=0$.
2. 求下列微分方程满足所给初始条件的特解:
(1) $\frac{\mathrm{d} y}{\mathrm{~d} x}-y \tan x=\sec x,\left.y\right|_{x=0}=0$;
(2) $\frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{y}{x}=\frac{\sin x}{x},\left.y\right|_{, x \rightarrow \pi}=1$;
(3) $\frac{\mathrm{d} y}{\mathrm{~d} x}+y \cot x=5 \mathrm{e}^{\operatorname{an} x},\left.y\right|_{x=\frac{x}{2}}=-4$;
(4) $\frac{\mathrm{d} y}{\mathrm{~d} x}+3 y=8,\left.y\right|_{x=0}=2$;
(5) $\frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{2-3 x^{2}}{x^{3}} y=1,\left.y\right|_{, \ldots 1}=0$.
3. 求一曲线的方程,这曲线通过原点,并且它在点 $(x, y)$ 处的切线斜率等于 $2 x+y$.
4. 设有一质垪为 $m$ 的质点作直线运动. 众速度等于笭的时刻起, 有一个与运动方向一 致、大小与时间成正比 (比例系数为 $k_{1}$ ) 的力作用于它, 此外还受一与速度成正比 (比例系数 为 $k_{2}$ ) 的阻力作用. 求质点运动的速度与时间的函数关系.
5. 设有一个由电阻 $R=10 \Omega$ 、电感 $L=2 \mathrm{H}$ 和电源电压 $E=20 \sin 5 t \mathrm{~V}$ 串联组成的电 路. 开关 S 合上后, 电路中有电流通过. 求电流 $i$ 与时间 $t$ 的函数关系.
6. 验证形如 $y f(x y) \mathrm{d} x+x g(x y) \mathrm{d} y=0$ 的微分方程, 可经变掣代换 $v=x y$ 化为可分离 变些的方程,并求其通解.
7. 用适当的变量代换将下列方程化为可分离变䑤的方程,然后求出通解：
(1) $\frac{\mathrm{d} y}{\mathrm{~d} x}=(x+y)^{2}$;
(2) $\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{x-y}+1$;
(3) $x y^{\prime}+y=y(\ln x+\ln y)$;
(4) $y^{\prime}=y^{2}+2(\sin x-1) y+\sin ^{2} x-2 \sin x-\cos x+1$;
(5) $y(x y+1) \mathrm{d} x+x\left(1+x y+x^{2} y^{2}\right) \mathrm{d} y=0$. -8. 求下列伯努利方程的通解:
(1) $\frac{\mathrm{d} y}{\mathrm{~d} x}+y=y^{2}(\cos x-\sin x)$;
(2) $\frac{\mathrm{d} y}{\mathrm{~d} x}-3 x y=x y^{2}$;
(3) $\frac{\mathrm{d} y}{\mathrm{~d} x}+\frac{1}{3} y=\frac{1}{3}(1-2 x) y^{4}$;
(4) $\frac{\mathrm{d} y}{\mathrm{~d} x}-y=x y^{5}$;
(5) $x \mathrm{~d} y-\left[y+x y^{3}(1+\ln x)\right] \mathrm{d} x=0$.

## 第五节 可降阶的高阶微分方程

从这一节起我们将讨论二阶及二阶以上的微分方程, 即所谓高阶微分方程. 对于有些高阶微分方程, 我们可以通过代换将它化成较低阶的方程来求解. 以二 阶微分方程

$$
y^{\prime \prime}=f\left(x, y, y^{\prime}\right)
$$

而论, 如果我们能设法作代换把它从二阶降至一阶, 那么就有可能应用前面几节 中所讲的方法来求出它的解了.

下面介绍三种容易降阶的高阶微分方程的求解方法.

## 一、 $y^{(n)}=f(x)$ 型的微分方程

微分方程

$$
y^{(x)}=f(x)
$$

的右端仅含有自变量 $x$. 容易看出, 只要把 $y^{(n-1)}$ 作为新的未知函数, 那么 (2) 式 就是新未知函数的一阶微分方程. 两边积分, 就得到一个 $n-1$ 阶的微分方程

$$
y^{(12-1)}=\int f(x) \mathrm{d} x+C_{1} .
$$

同理可得

$$
y^{(n-2)}=\int\left[\int f(x) \mathrm{d} x+C_{1}\right] \mathrm{d} x+C_{2} .
$$

依此法继续进行, 接连积分 $n$ 次, 便得方程 (2) 的含有 $n$ 个任意常数的通解.

例 1 求微分方程

$$
y^{\prime \prime \prime}=\mathrm{e}^{2 \cdot r}-\cos x
$$

的通解.

解 对所给方程接连积分三次, 得

$$
\begin{gathered}
y^{\prime \prime}=\frac{1}{2} \mathrm{e}^{2 x}-\sin x+C, \\
y^{\prime}=\frac{1}{4} \mathrm{e}^{2 \cdot x}+\cos x+C x+C_{2}, \\
y=\frac{1}{8} \mathrm{e}^{2 \cdot x}+\sin x+C_{1} x^{2}+C_{2} x+C_{3} \quad\left(C_{1}=\frac{C}{2}\right) .
\end{gathered}
$$

这就是所求的通解.

例 2 质量为 $m$ 的质点受力 $F$ 的作用沿 $O x$ 轴作直线运动. 设力 $F=F(t)$ 在 开始时刻 $t=0$ 时 $F(0)=F_{10}$, 随着时间 $t$ 的增大, 力 $F$ 均匀地减小, 直到 $t=T$ 时, $F(T)=0$. 如果开始时质点位于原点, 且初速度为零, 求这质点的运动规律.

解 设 $x=x(t)$ 表示在时刻 $t$ 时质点的位置, 根据牛顿第二定律, 质点运 动的微分方程为

$$
m \frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=F(t) .
$$

由题设, 力 $F(t)$ 随 $t$ 增大而均匀地减小, 且 $t=0$ 时, $F(0)=F_{0}$, 所以 $F(t)=$ $F_{0}-k t$; 又当 $t=T$ 时, $F(T)=0$, 从而

$$
F(t)=F_{0}\left(1-\frac{t}{T}\right) .
$$

于是方程 (3) 可以写成

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=\frac{F_{0}}{m}\left(1-\frac{t}{T}\right)
$$

其初始条件为

$$
\left.x\right|_{,=0}=0,\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=0}=0 .
$$

把(4)式两端积分, 得

即

$$
\begin{gathered}
\frac{\mathrm{d} x}{\mathrm{~d} t}=\frac{F_{0}}{m} \int\left(1-\frac{t}{T}\right) \mathrm{d} t, \\
\frac{\mathrm{d} x}{\mathrm{~d} t}=\frac{F_{0}}{m}\left(t-\frac{t^{2}}{2 T}\right)+C_{1} .
\end{gathered}
$$

将条件 $\left.\frac{\mathrm{d} x}{\mathrm{~d} \ell}\right|_{,=0}=0$ 代入 $(5)$ 式,得

$$
C_{1}=0 \text {, }
$$

于是(5)式成为

$$
\frac{\mathrm{d} x}{\mathrm{~d} t}=\frac{F_{0}}{m}\left(t-\frac{t^{2}}{2 T}\right)
$$

把(6)式两端积分,得

$$
x=\frac{F_{0}}{m}\left(\frac{t^{2}}{2}-\frac{t^{3}}{6 T}\right)+C_{2},
$$

将条件 $\left.x\right|_{\imath=\|}=0$ 代入上式, 得

$$
C_{2}=0 \text {. }
$$

于是所求质点的运动规律为

$$
x=\frac{F_{0}}{m}\left(\frac{t^{2}}{2}-\frac{t^{3}}{6 T}\right), 0 \leqslant t \leqslant T .
$$

## 二、 $y^{\prime \prime}=f\left(x, y^{\prime}\right)$ 型的微分方程

方程

$$
y^{\prime \prime}=f\left(x, y^{\prime}\right)
$$

的右端不显含未知函数 $y$. 如果我们设 $y^{\prime}=p$, 那么

$$
y^{\prime \prime}=\frac{\mathrm{d} p}{\mathrm{~d} x}=p^{\prime},
$$

而方程 (7) 就成为

$$
p^{\prime}=f(x, p) .
$$

这是一个关于变量 $x 、 p$ 的一阶微分方程. 设其通解为

$$
p=\varphi\left(x, C_{1}\right),
$$

但是 $p=\frac{\mathrm{d} y}{\mathrm{~d} x}$, 因此又得到一个一阶微分方程

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\varphi\left(x, C_{1}\right) .
$$

对它进行积分, 便得方程 (7)的通解为

$$
y=\int \varphi\left(x, C_{1}\right) \mathrm{d} x+C_{2} .
$$

例 3 求微分方程

$$
\left(1+x^{2}\right) y^{\prime \prime}=2 x y^{\prime}
$$

满足初始条件

$$
\left.y\right|_{x=0}=1,\left.\quad y^{\prime}\right|_{x=0}=3
$$

的特解.

解 所给方程是 $y^{\prime \prime}=f\left(x, y^{\prime}\right)$ 型的. 设 $y^{\prime}=p$, 代入方程并分离变量后, 有

$$
\frac{\mathrm{d} p}{p}=\frac{2 x}{1+x^{2}} \mathrm{~d} x \text {. }
$$

两端积分, 得

$$
\begin{gathered}
\ln |p|=\ln \left(1+x^{2}\right)+C, \\
\text { 即 } p=y^{\prime}=C_{1}\left(1+x^{2}\right) \quad\left(C_{1}= \pm \mathrm{e}^{c}\right) .
\end{gathered}
$$

由条件 $\left.y^{\prime}\right|_{x=0}=3$, 得

$$
C_{1} \stackrel{\circ}{=} 3,
$$

所以

$$
y^{\prime}=3\left(1+x^{2}\right) \text {. }
$$

两端再积分, 得

$$
y=x^{3}+3 x+C_{2} \text {. }
$$

又由条件 $\left.y\right|_{x=0}=1$, 得

$$
C_{2}=1 \text {, }
$$

于是所求的特解为

$$
y=x^{3}+3 x+1 .
$$

例 4 设有一均匀、柔软的绳索,两端固定,绳索仅受重力的作用而下垂. 试 问该绳索在平衡状态时是怎样的曲线?

解 设绳索的最低点为 $A$. 取 $y$ 轴通过点 $A$ 铅直向上,并取 $x$ 轴水平向右. 且 $|O A|$ 等于某个定值 (这个定值将在以后说明). 设绳索曲线的方程为 $y=$ $\varphi(x)$. 考察绳索上点 $A$ 到另一点 $M(x, y)$ 间的一段弧 $\overparen{A M}$, 设其长为 $s$. 假定绳 索的线密度为 $\rho$, 则弧 $\overparen{A M}$ 所受重力为 $\rho g s$. 由于绳索是柔软的, 因而在点 $A$ 处的 张力沿水平的切线方向, 其大小设为 $H$; 在点 $M$ 处的张力沿该点处的切线方 向,设其倾角为 $\theta$, 其大小为 $T$ (图 7-6). 因作用于弧段 $\overparen{A M}$ 的外力相互平衡, 把作用于弧 $\overparen{A M}$ 上的力沿铅直及水平两方向分解,得

$$
T \sin \theta=\rho g s, T \cos \theta=H .
$$

将此两式相除,得

$$
\tan \theta=\frac{1}{a} s \quad\left(a=\frac{H}{\rho g}\right) .
$$

由于 $\tan \theta=y^{\prime}, s=\int_{0}^{r} \sqrt{1+y^{\prime 2}} \mathrm{~d} x$, 代入上式即得

$$
y^{\prime}=\frac{1}{a} \int_{0}^{x} \sqrt{1+y^{\prime 2}} \mathrm{~d} x .
$$

将上式两端对 $x$ 求导, 便得 $y=\varphi(x)$ 满足的微分方程

$$
\text { - } y^{\prime \prime}=\frac{1}{a} \sqrt{1+y^{\prime 2}} .
$$

取原点 $O$ 到点 $A$ 的距离为定值 $a$, 即 $|O A|=a$, 那么初始条件为

$$
\left.y\right|_{. r=0}=a,\left.y^{\prime}\right|_{. r=0}=0 \text {. }
$$

下面来解方程 (8).

方程 $(8)$ 属于 $y^{\prime \prime}=f\left(x, y^{\prime}\right)$ 的类型. 设

$$
y^{\prime}=p \text {, 则 } y^{\prime \prime}=\frac{\mathrm{d} p}{\mathrm{~d} x} \text {, }
$$

代入方程 $(8)$,并分离变量,得

$$
\frac{\mathrm{d} p}{\sqrt{1+p^{2}}}=\frac{\mathrm{d} x}{a}
$$

两端积分,得

$$
\ln \left(p+\sqrt{1+p^{2}}\right)=\frac{x}{a}+C_{1} .
$$

把条件 $\left.y^{\prime}\right|_{x=0}=\left.p\right|_{. r=0}=0$ 代入(9)式,得

$$
C_{1}=0 \text {, }
$$

于是(9)式成为

解得

$$
\begin{gathered}
\ln \left(p+\sqrt{1+p^{2}}\right)=\frac{x}{a}, \\
p=\frac{1}{2}\left(\mathrm{e}^{\frac{x}{u}}-\mathrm{e}^{-\frac{x}{a}}\right), \\
y^{\prime}=\frac{1}{2}\left(\mathrm{e}^{\frac{1}{a}}-\mathrm{e}^{-\frac{x}{a}}\right) .
\end{gathered}
$$

积分上式两端,便得

$$
y=\frac{a}{2}\left(\mathrm{e}^{\frac{a}{a}}+\mathrm{e}^{-\frac{x}{a}}\right)+C_{2} .
$$

将条件 $\left.y\right|_{x=0}=a$ 代入 $(10)$ 式, 得

$$
C_{2}=0 \text {. }
$$

于是该绳索的形状可由曲线方程

$$
y=\frac{a}{2}\left(\mathrm{e}^{\frac{x}{a}}+\mathrm{e}^{-\frac{x}{a}}\right)
$$

来表示. 这曲线叫做悬链线.

## 三、 $y^{\prime \prime}=f\left(y, y^{\prime}\right)$ 型的微分方程

方程

$$
y^{\prime \prime}=f\left(y, y^{\prime}\right)
$$

中不明显地含自变量 $x$. 为了求出它的解. 我们令 $y^{\prime}=p$, 并利用复合函数的求 导法则把 $y^{\prime \prime}$ 化为对 $y$ 的导数, 即

$$
y^{\prime \prime}=\frac{\mathrm{d} \rho}{\mathrm{d} x}=\frac{\mathrm{d} \rho}{\mathrm{d} y} \cdot \frac{\mathrm{d} y}{\mathrm{~d} x}=p \frac{\mathrm{d} \rho}{\mathrm{d} y} .
$$

这样, 方程 (11) 就成为

$$
p \frac{\mathrm{d} p}{\mathrm{~d} y}=f(y, p) .
$$

这是一个关于变量 $y, p$ 的一阶微分方程. 设它的通解为

$$
y^{\prime}=p=\varphi\left(y, C_{1}\right),
$$

分离变量并积分,便得方程 (11) 的通解为

$$
\int \frac{\mathrm{d} y}{\varphi\left(y, C_{1}\right)}=x+C_{2} \text {. }
$$

例 5 求微分方程

$$
y y^{\prime \prime}-y^{\prime 2}=0
$$

的通解.

解 方程 (12)不明显地含自变量 $x$, 设

代入方程 (12), 得

$$
y^{\prime}=p \text {, 则 } y^{\prime \prime}=p \frac{\mathrm{d} p}{\mathrm{~d} y} \text {, }
$$

$$
y p \frac{\mathrm{d} p}{\mathrm{~d} y}-p^{2}=0 .
$$

在 $y \neq 0 、 p \neq 0$ 时, 约去 $p$ 并分离变量,得

$$
\frac{\mathrm{d} p}{p}=\frac{\mathrm{d} y}{y} \text {. }
$$

两端积分, 得

$$
\ln |p|=\ln |y|+C,
$$

即

$$
p=C_{1} y \text {.或 } y^{\prime}=C_{1} y \quad\left(C_{1}= \pm \mathrm{e}^{c^{c}}\right) .
$$

再分离变量并两端积分, 便得方程 (12)的通解为

$$
\ln |y|=C_{1} x+C_{2}^{\prime},
$$

或

$$
y=c_{2} \mathrm{e}^{r_{1} \cdot r} \quad\left(C_{2}= \pm \mathrm{e}^{\left(r_{2}\right)}\right. \text {. }
$$

例 6 一个离地面很高的物体, 受地球引力的作用由静止开始落向地面. 求 它落到地面时的速度和所需的时间 (不计空气阻力).

解 取联结地球中心与该物体的直线为 $y$ 轴, 其方向铅直向上, 取地球的 中心为原点 $O$ (图 7-7).

设地球的半径为 $R$,物体的质量为 $m$, 物体开始下落。 时与地球中心的距离为 $l(l>R)$, 在时刻 $l$ 物体所在位 置为 $y=\varphi(t)$, 于是速度为 $v(t)=\frac{\mathrm{d} y}{\mathrm{~d} t}$. 根据万有引力定 律, 即得微分方程

$$
\begin{aligned}
m \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}} & =-\frac{G m M}{y^{2}}, \\
\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}} & =-\frac{G M}{y^{2}},
\end{aligned}
$$

其中 $M$ 为地球的质量, $G$ 为引力常数. 因为当 $y=R$ 时,

$$
\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}=-\frac{g R^{2}}{y^{2}} .
$$

初始条件是

$$
\left.y\right|_{t=0}=l,\left.y^{\prime}\right|_{t-0}=0 \text {. }
$$

先求物体到达地面时的速度. 由 $\frac{\mathrm{d} y}{\mathrm{~d} t}=v$, 得

$$
\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}=\frac{\mathrm{d} v}{\mathrm{~d} t}=\frac{\mathrm{d} v}{\mathrm{~d} y} \cdot \frac{\mathrm{d} y}{\mathrm{~d} t}=v \frac{\mathrm{d} v}{\mathrm{~d} y},
$$

代入方程 (14) 并分离变量, 得

$$
v \mathrm{~d} v=-\frac{g R^{2}}{y^{2}} \mathrm{~d} y .
$$

两端积分, 得

$$
v^{2}=\frac{2 g R^{2}}{y}+C_{1} \text {. }
$$

把初始条件代入上式,得

$$
C_{1}=-\frac{2 g R^{2}}{l}
$$

于是

$$
v^{2}=2 g R^{2}\left(\frac{1}{y}-\frac{1}{l}\right), v=-R \sqrt{2 g\left(\frac{1}{y}-\frac{1}{l}\right)} .
$$

这里取负号是由于物体运动的方向与 $y$ 轴的正向相反的缘故.

在 (15) 式中令 $y=R$, 就得到物体到达地面时的速度为

$$
v=-\sqrt{\frac{2 g R(l-R)}{l}} .
$$

下面来求物体落到地面所需的时间.由(15)式有

$$
\frac{\mathrm{d} y}{\mathrm{~d} t}=v=-R \sqrt{2 g\left(\frac{1}{y}-\frac{1}{l}\right)},
$$

分离变量得

$$
\mathrm{d} t=-\frac{1}{R} \sqrt{\frac{l}{2 g}} \sqrt{\frac{y}{l-y}} \mathrm{~d} y .
$$

两端积分 (对右端积分利用置换 $y=l \cos ^{2} u$ ), 得

$$
t=\frac{1}{R} \sqrt{\frac{l}{2 g}}\left(\sqrt{l y-y^{2}}+l \arccos \sqrt{\frac{y}{l}}\right)+C_{2} .
$$

由条件 $\left.y\right|_{1,0}=l$, 得

$$
C_{2}=0
$$

于是(16)式成为

$$
t=\frac{1}{R} \sqrt{\frac{l}{2 g}}\left(\sqrt{l y-y^{2}}+l \arccos \sqrt{\frac{y}{l}}\right) .
$$

在上式中令 $y=R$, 便得到物体到达地面所需的时间为

$$
t=\frac{1}{R} \sqrt{\frac{l}{2 g}}\left(\sqrt{l R-R^{2}}+l \arccos \sqrt{\frac{R}{l}}\right)
$$

## 习 题 7-5

1. 求下列各微分方程的近解:
(1) $y^{\prime \prime}=x+\sin x$;
(2) $y^{\prime \prime \prime}=x \mathrm{e}^{\prime \prime}$;
(3) $y^{\prime \prime}=\frac{1}{1+x^{2}}$;
(4) $y^{\prime \prime}=1+y^{\prime 2}$;
(5) $y^{\prime \prime}=y^{\prime}+x$;
(6) $x y^{\prime \prime}+y^{\prime}=0$;
(7) $y y^{\prime \prime}+2 y^{\prime 2}=0$;
(8) $y^{\prime \prime} y^{\prime \prime}-1=0$;
(9) $y^{\prime \prime}=\frac{1}{\sqrt{y}}$;
(10) $y^{\prime \prime}=\left(y^{\prime}\right)^{3}+y^{\prime}$.
2. 求下列各微分方程满足所给初始条件的特解:
(1) $y^{\prime \prime} y^{\prime \prime}+1=0,\left.y\right|_{, x-1}=1,\left.y^{\prime}\right|_{x-1}=0$;
(2) $y^{\prime \prime}-a y^{\prime 2}=0,\left.y\right|_{, \ldots 0}=0,\left.y^{\prime}\right|_{x=11}=-1$;
(3) $y^{\prime \prime \prime}=\mathrm{e}^{a x},\left.y\right|_{, r=1}=\left.y^{\prime}\right|_{1,1}=\left.y^{\prime \prime}\right|_{, \ldots 1}=0$;
(4) $y^{\prime \prime}=\mathrm{e}^{2, y},\left.y\right|_{,+10}=\left.y^{\prime}\right|_{1, \ldots}=0$;
(5) $y^{\prime \prime}=3 \sqrt{y},\left.y\right|_{, x=0}=1,\left.y^{\prime}\right|_{x=0}=2$;
(6) $y^{\prime \prime}+\left(y^{\prime}\right)^{2}=1,\left.y\right|_{x=0}=0,\left.y^{\prime}\right|_{1=1}=0$.
3. 试求 $y^{\prime \prime}=x$ 的经过点 $M(0,1)$ 且在此点与直线 $y=\frac{x}{2}+1$ 相切的积分曲线.
4. 设有一质量为 $m$ 的物体, 在空中由静.上开始下落, 如果空气阻力为 $R=c v$ (其中 $c$ 为 常数, $v$ 为物体运动的速度), 试求物体下落的距离 $s$ 与时间 $t$ 的函数关系.

## 第六节 高阶线性微分方程

本节和以下两节, 我们将讨论在实际问题中应用得较多的所谓高阶线性微 分方程. 讨论时以二阶线性微分方程为主.

## 一、二阶线性微分方程举例

例 1 设有一个弹簧, 它的上端固定, 下端挂一个质量为 $m$ 的物体. 当物体 处于静止状态时, 作用在物体上的重力与弹性力大小相等、方向相反. 这个位置 就是物体的平衡位置. 如图 7-8, 取 $x$ 轴铅直向下, 并取物体的平衡位置为坐标 原点.

如果使物体具有一个初始速度 $v_{11} \neq 0$, 那么物体便离开平衡位置, 并在平衡 位置附近作上下振动. 在振动过程中,物体的位置 $x$ 随时间 $\iota$ 变 化, 即 $x$ 是 $t$ 的函数: $x=x(t)$. 要确定物体的振动规律, 就要求出 函数 $x=x(\iota)$.

由力学知道, 弹䈠使物体回到平衡位置的弹性恢复力 $f($ 它 不他括在平衡位置时和重力 $m g$ 相平衡的那一部分弹性力) 和物 体离开平衡位置的位移 $x$ 成丑比:

$$
f=-c x \text {, }
$$

其中 $c$ 为弹簧的弹性系数, 负号表示弹性恢复力的方向和物体位 移的方向相反.

另外,物体在运动过程中还受到阻尼介质(如空气、油等)的阻

$$
R=-\mu \frac{\mathrm{d} x}{\mathrm{~d} t} .
$$

根据上述关于物体受力情况的分析，由牛顿第二定律得

$$
m \frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=-c x-\mu \frac{\mathrm{d} x}{\mathrm{~d} t} .
$$

移项，并记

$$
2 n=\frac{\mu}{m}, k^{2}=\frac{c}{m} \text {, }
$$

则上式化为

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+2 n \frac{\mathrm{d} x}{\mathrm{~d} t}+k^{2} x=0 .
$$

这就是在有阻尼的情况下,物体自由振动的微分方程。

如果物体在振动过程中, 还受到铅直干扰力

$$
F=H \sin p t
$$

的作用, 则有

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+2 n \frac{\mathrm{d} x}{\mathrm{~d} t}+k^{2} x=h \sin \cdot p t
$$

其中 $h=\frac{H}{m}$. 这就是强迫振动的微分方程.

例 2 设有一个由电阻 $R$ 、自感 $L$ 、电容 $C$ 和电源 $E$ 串联组成的电路, 其中 $R 、 L$ 及 $C$ 为常数, $E=E_{\mathrm{m}} \sin \omega t$, 这里 $E_{\mathrm{m}}$ 及 $\omega$ 也是常数 (图 7-9).

设电路中的电流为 $i(\iota)$, 电容器极板上的电荷量为 $q(l)$, 两极板间的电压 为 $u_{c}$, 自感电动势为 $E_{\mathrm{L}}$. 由电学知道

$$
i=\frac{\mathrm{d} q}{\mathrm{~d} t}, u_{C}=\frac{q}{C}, E_{\mathrm{L}}=-L \frac{\mathrm{d} i}{\mathrm{~d} t},
$$

根据回路电压定律, 得

$$
E-L \frac{\mathrm{d} i}{\mathrm{~d} t}-\frac{q}{C}-R i=0,
$$

即

$$
L C \frac{\mathrm{d}^{2} u_{C}}{\mathrm{~d} t^{2}}+R C \frac{\mathrm{d} u_{c}}{\mathrm{~d} t}+u_{\mathrm{c}}=E_{\mathrm{n}} \sin \omega t,
$$

或写成

$$
\frac{\mathrm{d}^{2} u_{c}}{\mathrm{~d} t^{2}}+2 \beta \frac{\mathrm{d} u_{c}}{\mathrm{~d} t}+\omega_{0}^{2} u_{\mathrm{C}}=\frac{E_{\mathrm{m}}}{L C} \sin \omega t .
$$

式中 $\beta=\frac{R}{2 L}, \omega_{0}=\frac{1}{\sqrt{L C}}$. 这就是串联电路的振渼方程.

如果电容器经充电后撤去外电源 $(E=0)$, 则方程 (3) 成为

$$
\frac{\mathrm{d}^{2} u_{\mathrm{c}}}{\mathrm{d} t^{2}}+2 \beta \frac{\mathrm{d} u_{\mathrm{c}}}{\mathrm{d} t}+\omega_{11}^{2} u_{\mathrm{c}}=0 .
$$

例 1 和例 2 虽然是两个不同的实际问题,但是仔细观察一下所得出的方程 (2)和 (3), 就会发现它们可以归结为同一个形式

$$
\frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}+P(x) \frac{\mathrm{d} y}{\mathrm{~d} x}+Q(x) y=f(x),
$$

而方程 (1) 和方程 (4) 都是方程 (5) 的特殊情形: $f(x) \equiv 0$. 在工程技术的其他许多 问题中，也会遇到上述类型的微分方程.

方程 (5) 叫做二阶线性微分方程. 当方程右端 $f(x) \equiv 0$ 时,方程叫做齐次的; 当 $f(x) \neq 0$ 时,方程叫做非齐次的.

于是方程 (2)、(3)都是二阶非齐次线性微分方程; 方程 (1)、(4) 都是二阶齐 次线性微分方程.

要进一步讨论例 1 和例 2 中的问题, 就需要解二阶线性微分方程. 为此, 下 面来讨论二阶线性微分方程的解的一些性质,这些性质可以推广到 $n$ 阶线性方 程

$$
y^{(n)}+a_{1}(x) y^{(n-1)}+\cdots+a_{n-1}(x) y^{\prime}+a_{n}(x) y=f(x) \text {. }
$$

## 二、线性微分方程的解的结构

先讨论二阶齐次线性方程

$$
y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=0 .
$$

定理 1 如果函数 $y_{1}(x)$ 与 $y_{2}(x)$ 是方程 $(6)$ 的两个解，那么

$$
y=C_{1} y_{1}(x)+C_{2} y_{2}(x)
$$

也是(6)的解,其中 $C_{1} 、 C_{2}$ 是任意常数.

$$
\begin{aligned}
& \text { 证 将(7)式代入 (6) 式左端, 得 } \\
& {\left[C_{1} y_{1}^{\prime \prime}+C_{2} y_{2}^{\prime \prime}\right]+P(x)\left[C_{1} y_{1}^{\prime}+C_{2} y_{2}^{\prime}\right]+Q(x)\left[C_{1} y_{1}+C_{2} y_{2}\right] } \\
= & C_{1}\left[y_{1}^{\prime \prime}+P(x) y_{1}^{\prime}+Q(x) y_{1}\right]+C_{2}\left[y_{2}^{\prime \prime}+P(x) y_{2}^{\prime}+Q(x) y_{2}\right] .
\end{aligned}
$$

由于 $y_{1}$ 与 $y_{2}$ 是方程(6)的解, 上式右端方括号中的表达式都恒等于笭, 因而整 个式子恒等于零,所以 (7)式是方程 (6) 的解.

解 (7) 从形式上来看含有 $C_{1}$ 与 $C_{2}$ 两个任意常数,但它不一定是方程 (6) 的 通解. 例如, 设 $y_{1}(x)$ 是 (6) 的一个解, 则 $y_{2}(x)=2 y_{1}(x)$ 也是 (6) 的解. 这时 (7) 式成为 $y=C_{1} y_{1}(x)+2 C_{2} y_{1}(x)$, 可以把它改写成 $y=C y_{1}(x)$, 其中 $C=C_{1}+$ $2 C_{2}$. 这显然不是 (6) 的通解. 那么在什么情况下 (7) 式才是方程 (6) 的通解呢? 要解决这个问题, 还得引人一个新的概念, 即所谓函数组的线性相关与线性无 关.

设 $y_{1}(x), y_{2}(x), \cdots, y_{n}(x)$ 为定义在区间 $I$ 上的 $n$ 个函数, 如果存在 $n$ 个 不全为零的常数 $k_{1}, k_{2}, \cdots, k_{n}$, 使得当 $x \in I$ 时有恒等式

$$
k_{1} y_{1}+k_{2} y_{2}+\cdots+k_{n} y_{n} \equiv 0
$$

成立,那么称这 $n$ 个函数在区间 $I$ 上线性想关; 否则称线性无关.

例如, 函数 $1, \cos ^{2} x, \sin ^{2} x$ 在整个数轴上是线性相关的. 因为取 $k_{1}=1$, $k_{2}=k_{3}=-1$, 就有恒等式

$$
1-\cos ^{2} x-\sin ^{2} x \equiv 0 .
$$

又如, 函数 $1, x, x^{2}$ 在任何区间 $(a, b)$ 内是线性无关的. 因为如果 $k_{1}, k_{2}, k_{3}$ 不 全为零,那么在该区间内至多只有两个 $x$ 值能使二次三项式

$$
k_{1}+k_{2} x+k_{3} x^{2}
$$

为零; 要使它恒等于零, 必须 $k_{1}, k_{2}, k_{3}$ 全为零.

应用上述概念可知,对于两个函数的情形, 它们线性相关与否, 只要看它们 的比是否为常数: 如果比为常数, 那么它们就线性相关; 否则就线性无关.

有了一组函数线性相关或线性无关的概念后, 我们有如下关于二阶齐次线 性微分方程 (6) 的通解结构的定理.

定理 2 如果 $y_{1}(x)$ 与 $y_{2}(x)$ 是方程 (6) 的两个线性无关的特解, 那么

$$
y=C_{1} y_{1}(x)+C_{2} y_{2}(x) \quad\left(C_{1} 、 C_{2}\right. \text { 是任意常数) }
$$

## 就是方程 (6) 的通解.

例如,方程 $y^{\prime \prime}+y=0$ 是二阶齐次线性方程（这里 $p(x) \equiv 0, Q(x) \equiv 1$ ). 容易 验证, $y_{1}=\cos x$ 与 $y_{2}=\sin x$ 是所给方程的两个解, 且 $\frac{y_{2}}{y_{1}}=\frac{\sin x}{\cos x}=\tan x \neq$ 常数, 即它们是线性无关的. 因此方程 $y^{\prime \prime}+y=0$ 的通解为

$$
y=C_{1} \cos x+C_{2} \sin x .
$$

又如,方程 $(x-1) y^{\prime \prime}-x y^{\prime}+y=0$ 也是二阶齐次线性方程 (这里 $\left.P(x)=-\frac{x}{x-1}, Q(x)=\frac{1}{x-1}\right)$. 容易验证 $y_{1}=x, y_{2}=\mathrm{e}^{x}$ 是所给方程的两个 解, 且 $\frac{y_{2}}{y_{1}}=\frac{\mathrm{e}^{x}}{x} \not$ 常数, 即它们是线性无关的. 因此方程的通解为

$$
y=C_{1} x+C_{2} \mathrm{e}^{x} .
$$

定理 2 不难推广到 $n$ 阶齐次线性方程.

推论 如果 $y_{1}(x), y_{2}(x), \cdots, y_{n}(x)$ 是 $n$ 阶齐次线性方程

$$
y^{(n)}+a_{1}(x) y^{(n-1)}+\cdots+a_{n-1}(x) y^{\prime}+a_{n}(x) y=0
$$

的 $n$ 个线性无关的解, 那么, 此方程的通解为

$$
y=C_{1} y_{1}(x)+C_{2} y_{2}(x)+\cdots+C_{n} y_{n}(x),
$$

其中 $C_{1}, C_{2}, \cdots, C_{n}$ 为任意常数.

下面讨论二阶非齐次线性方程 (5), 我们把方程 (6) 叫做与非齐次方程(5)对 应的齐次方程.

在第四节中我们已经看到,一阶非齐次线性微分方程的通解由两部分构成: 一部分是对应的齐次方程的通解; 另一部分是非齐次方程本身的一个特解. 实际 上,不仅一阶非齐次线性微分方程的通解具有这样的结构, 而且二阶及更高阶的 非齐次线性微分方程的通解也具有同样的结构.

定理 3 设 $y^{*}(x)$ 是二阶非齐次线性方程

$$
y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=f(x)
$$

的一个特解. $Y(x)$ 是与 (5) 对应的齐次方程 (6) 的通解, 那么

$$
y=Y(x)+y^{*}(x)
$$

## 是二阶非齐次线性微分方程 (5) 的通解.

证 把 (8)式代入方程 (5) 的左端,得

$$
\begin{aligned}
& \left(Y^{\prime \prime}+y^{\prime \prime \prime}\right)+P(x)\left(Y^{\prime}+y^{\prime \prime}\right)+Q(x)\left(Y+y^{*}\right) \\
= & {\left[Y^{\prime \prime}+P(x) Y^{\prime}+Q(x) Y\right]+\left[y^{\prime \prime}+P(x) y^{\prime \prime}+Q(x) y^{*}\right], }
\end{aligned}
$$

由于 $Y$ 是方程 (6) 的解, $y^{*}$ 是 (5) 的解, 可知第一个括号内的表达式恒等于零, 第二个恒等于 $f(x)$. 这样, $y=Y+y^{*}$ 使 (5) 的两端恒等. 即 (8) 式是方程 (5) 的 解.

由于对应的齐次方程 (6) 的通解 $Y=C_{1} y_{1}+C_{2} y_{2}$ 中含有两个任意常数,所 以 $y=Y+y^{*}$ 中也含有两个任意常数, 从而它就是二阶非齐次线性方程 (5) 的 通解.

例如,方程 $y^{\prime \prime}+y=x^{2}$ 是二阶非齐次线性微分方程. 已知 $Y=C_{1} \cos x+$ $C_{2} \sin x$ 是对应的齐次方程 $y^{\prime \prime}+y=0$ 的通解; 又容易验证 $y^{*}=x^{2}-2$ 是所给方 程的一个特解. 因此

$$
y=C_{1} \cos x+C_{2} \sin x+x^{2}-2
$$

是所给方程的通解.

非齐次线性微分方程 (5) 的特解有时可用下述定理来帮助求出.

定理 4 设非齐次线性方程 (5) 的右端 $f(x)$ 是两个函数之和, 即

$$
y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=f_{1}(x)+f_{2}(x),
$$

而 $y_{1}^{*}(x)$ 与 $y_{2}^{*}(x)$ 分别是方程

$$
\text { 与 } \quad \begin{aligned}
& y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=f_{1}(x) \\
& y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=f_{2}(x)
\end{aligned}
$$

的特解，那么 $y_{i}^{*}(x)+y_{2}^{*}(x)$ 就是原方程的特解.

证 将 $y=y_{1}^{*}+y_{2}^{*}$ 代入方程(9) 的左端, 得

$$
\begin{aligned}
& \left(y_{1}^{\prime}+y_{2}^{\prime}\right)^{\prime \prime}+P(x)\left(y_{1}^{\prime}+y_{2}^{*}\right)^{\prime}+Q(x)\left(y_{1}^{*}+y_{2}^{\prime}\right) \\
= & {\left[y_{1}^{\prime \prime \prime}+P(x) y_{1}^{\prime \prime}+Q(x) y_{1}^{\prime}\right]+\left[y_{2}^{\prime \prime \prime}+P(x) y_{2}^{\prime \prime}+Q(x) y_{2}^{\prime}\right] } \\
= & f_{1}(x)+f_{2}(x) .
\end{aligned}
$$

因此 $y_{i}^{*}+y_{2}^{*}$ 是方程 (9) 的一个特解.

这一定理通常称为线性微分方程的解的叠加原理.

定理 3 和定理 4 也可推广到 $n$ 阶非齐次线性方程,这里不再整述.

## *三、常数变易法

在第四节中, 为解一阶非齐次线性方程, 我们用了常数变易法. 这方法的特 点是: 如果 $C y_{1}(x)$ 是齐次线性方程的通解, 那么, 可以利用变换 $y=u y_{1}(x)$ (这 变换是把齐次方程的通解中的任意常数 $C$ 换成未知函数 $u(x)$ 而得到的)去解 非齐次线性方程. 这一方法也适用于解高阶线性方程.下面就二阶线性方程来作 讨论.

如果已知齐次方程 (6) 的通解为

$$
Y(x)=C_{1} y_{1}(x)+C_{2} y_{2}(x),
$$

那么, 可以用如下的常数变易法去求非齐次方程 (5) 的通解:

令

$$
y=y_{1}(x) v_{1}+y_{2}(x) v_{2},
$$

要确定未知函数 $v_{1}(x)$ 及 $v_{2}(x)$ 使 $(10)$ 式所表示的函数满足非齐次方程 $(5)$. 为 此, 对 (10)式求导, 得

$$
y^{\prime}=y_{1} v_{1}^{\prime}+y_{2} v_{2}^{\prime}+y_{1}^{\prime} v_{1}+y_{2}^{\prime} v_{2} \text {. }
$$

由于两个未知函数 $v_{1} 、 v_{2}$ 只需使 (10) 式所表示的函数满足一个关系式 (5), 所 以可规定它们再满足一个关系式. 从 $y^{\prime}$ 的上述表示式可看出, 为了使 $y^{\prime \prime}$ 的表示 式中不含 $v_{1}^{\prime \prime}$ 和 $v_{2}^{\prime \prime}$, 可设

从而

$$
y_{1} v_{1}^{\prime}+y_{2} v_{2}^{\prime}=0 \text {, }
$$

再求导, 得

$$
y^{\prime}=y_{1}^{\prime} v_{1}+y_{2}^{\prime} v_{2} \text {, }
$$

$$
y^{\prime \prime}=y_{1}^{\prime} v_{1}^{\prime}+y_{2}^{\prime} v_{2}^{\prime}+y_{1}^{\prime \prime} v_{1}+y_{2}^{\prime \prime} v_{2} .
$$

把 $y, y^{\prime} 、 y^{\prime \prime}$ 代入方程 (5), 得

$$
y_{1}^{\prime} v_{1}^{\prime}+y_{2}^{\prime} v_{2}^{\prime}+y_{1}^{\prime \prime} v_{1}+y_{2}^{\prime \prime} v_{2}+P\left(y_{1}^{\prime} v_{1}+y_{2}^{\prime} v_{2}\right)+Q\left(y_{1} v_{1}+y_{2} v_{2}\right)=f,
$$

整理得

$$
y_{1}^{\prime} v_{1}^{\prime}+y_{2}^{\prime} v_{2}^{\prime}+\left(y_{1}^{\prime \prime}+P y_{1}^{\prime}+Q y_{1}\right) v_{1}+\left(y_{2}^{\prime \prime}+P y_{2}^{\prime}+Q y_{2}\right) v_{2}=f .
$$

注意到 $y_{1}$ 及 $y_{2}$ 是齐次方程 (6) 的解, 故上式即为

$$
y_{1}^{\prime} v_{1}^{\prime}+y_{2}^{\prime} v_{2}^{\prime}=f \text {. }
$$

联立方程 (11) 与 $(12)$,在系数行列式

$$
W=\left|\begin{array}{ll}
y_{1} & y_{2} \\
y_{1}^{\prime} & y_{2}^{\prime}
\end{array}\right|=y_{1} y_{2}^{\prime}-y_{1}^{\prime} y_{2} \neq 0
$$

时,可解得

$$
v_{1}^{\prime}=-\frac{y_{2} f}{W}, v_{2}^{\prime}=\frac{y_{1} f}{W} .
$$

对上两式积分（假定 $f(x)$ 连续），得

$$
v_{1}=C_{1}+\int\left(-\frac{y_{2} f}{W}\right) \mathrm{d} x, v_{2}=C_{2}+\int \frac{y_{1} f}{W} \mathrm{~d} x .
$$

于是得非齐次方程 (5)的通解为

$$
y=C_{1} y_{1}+C_{2} y_{2}-y_{1} \int \frac{y_{2} f}{W} \mathrm{~d} x+y_{2} \int \frac{y_{1} f}{W} \mathrm{~d} x .
$$

例 3 已知齐次方程 $(x-1) y^{\prime \prime}-x y^{\prime}+y=0$ 的通解为 $Y(x)=C_{1} x+$ $C_{2} \mathrm{e}^{x}$, 求非齐次方程 $(x-1) y^{\prime \prime}-x y^{\prime}+y=(x-1)^{2}$ 的通解.

解 把所给方程写成标准形式

$$
y^{\prime \prime}-\frac{x}{x-1} y^{\prime}+\frac{1}{x-1} y=x-1 .
$$

令 $y=x v_{1}+\mathrm{e}^{x} v_{2}$. 按照

$$
\left\{\begin{array}{l}
y_{1} v_{1}^{\prime}+y_{2} v_{2}^{\prime}=0 \\
y_{1}^{\prime} v_{1}^{\prime}+y_{2}^{\prime} v_{2}^{\prime}=f .
\end{array}\right.
$$

有

$$
\left\{\begin{array}{l}
x v_{1}^{\prime}+\mathrm{e}^{\prime} v_{2}^{\prime}=0, \\
v_{1}^{\prime}+\mathrm{e}^{\prime} v_{2}^{\prime}=x-1,
\end{array}\right.
$$

解得

$$
v_{1}^{\prime}=-1, v_{2}^{\prime}=x \mathrm{e}^{-\cdot r} \text {. }
$$

积分，得

$$
v_{1}=C_{1}-x, v_{2}=C_{2}-(x+1) \mathrm{e}^{-r} .
$$

于是所求非齐次方程的通解为

$$
y=C_{1} x+C_{2} \mathrm{e}^{x}-\left(x^{2}+x+1\right) .
$$

如果只知齐次方程 $(6)$ 的一个不恒为零的解 $y_{1}(x)$, 那么, 利用变换 $y=$ $u y_{1}(x)$, 可把非齐次方程 (5) 化为一阶线性方程.

事实上,把

$$
y=y_{1} u, y^{\prime}=y_{1} u^{\prime}+y_{1}^{\prime} u, y^{\prime \prime}=y_{1} u^{\prime \prime}+2 y_{1}^{\prime} u^{\prime}+y_{1}^{\prime \prime} u
$$

代入方程 $(5)$, 得

$$
y_{1} u^{\prime \prime}+2 y_{1}^{\prime} u^{\prime}+y_{1}^{\prime \prime} u+P\left(y_{1} u^{\prime}+y_{1}^{\prime} u\right)+Q y_{1} u=f,
$$

即

$$
y_{1} u^{\prime \prime}+\left(2 y_{1}^{\prime}+P y_{1}\right) u^{\prime}+\left(y_{1}^{\prime \prime}+P y_{1}^{\prime}+Q y_{1}\right) u=f \text {. }
$$

由于 $y_{1}^{\prime \prime}+P y_{1}^{\prime}+Q y_{1} \equiv 0$, 故上式为

$$
y_{1} u^{\prime \prime}+\left(2 y_{1}^{\prime}+P y_{1}\right) u^{\prime}=f .
$$

令 $u^{\prime}=z$, 上式即化为一阶线性方程

$$
y_{1} z^{\prime}+\left(2 y_{1}^{\prime}+P y_{1}\right) z=f .
$$

把方程 (5)化为方程 (13)以后, 按一阶线性方程的解法, 设求得方程 (13) 的 通解为

$$
z=C_{2} Z(x)+z^{*}(x),
$$

积分得 $u=C_{1}+C_{2} U(x)+u^{*}(x)$ (其中 $U^{\prime}(x)=Z(x), u^{\prime \prime}(x)=z^{*}(x)$ ), 上式乘以 $y_{1}(x)$, 便得方程 (5) 的通解

$$
y=C_{1} y_{1}(x)+C_{2} U(x) y_{1}(x)+u^{*}(x) y_{1}(x) .
$$

上述方法显然也适用于求齐次方程 (6) 的通解.

例 4 已知 $y_{1}(x)=\mathrm{e}^{x}$ 是齐次方程 $y^{\prime \prime}-2 y^{\prime}+y=0$ 的解, 求非齐次方程 $y^{\prime \prime}-2 y^{\prime}+y=\frac{1}{x} \mathrm{e}^{x}$ 的通解.

解 令 $y=\mathrm{e}^{-x} u$, 则 $y^{\prime}=\mathrm{e}^{x}\left(u^{\prime}+u\right), y^{\prime \prime}=\mathrm{e}^{x}\left(u^{\prime \prime}+2 u^{\prime}+u\right)$, 代入非齐次方 程,得

$$
\mathrm{e}^{x}\left(u^{\prime \prime}+2 u^{\prime}+u\right)-2 \mathrm{e}^{x}\left(u^{\prime}+u\right)+\mathrm{e}^{x} u=\frac{1}{x} \mathrm{e}^{x},
$$

即

$$
\mathrm{e}^{x} u^{\prime \prime}=\frac{1}{x} \mathrm{e}^{x}, u^{\prime \prime}=\frac{1}{x} .
$$

这里不需再作变换去化为一阶线性方程, 只要直接积分, 便得

$$
u^{\prime}=C+\ln |x| \text {, }
$$

再积分得

即

$$
\begin{gathered}
u=C_{1}+C x+x \ln |x|-x, \\
u=C_{1}+C_{2} x+x \ln |x|\left(C_{2}=C-1\right) .
\end{gathered}
$$

## 于是所求通解为

$$
y=C_{1} \mathrm{e}^{r}+C_{2} x \mathrm{e}^{-r}+x \mathrm{e}^{x} \ln |x| .
$$

## 习 题 7-6

## 1. 下列函数组在其定义区间内哪些是线性无关的?

(1) $x, x^{2}$;
(2) $x, 2 . x$;
(3) $\mathrm{e}^{2 \cdot r}, 3 \mathrm{e}^{2 r}$.
(4) $\mathrm{e}^{-1}, \mathrm{e}^{\mathrm{1}}$;
(5) $\cos 2 x, \sin 2 x$;
(6) $\mathrm{e}^{x^{2}}, x \mathrm{e}^{x^{2}}$;
(7) $\sin 2 x, \cos x \sin x$;
(8) $\mathrm{e}^{x} \cos 2 x, \mathrm{e}^{x} \sin 2 x$;
(9) $\ln x, x \ln x$;
(10) $\mathrm{e}^{a \cdot r}, \mathrm{e}^{\mathrm{bx}}(a \neq b)$.

2. 验证 $y_{1}=\cos \omega x$ 及 $y_{2}=\sin \omega x$ 都是方程 $y^{\prime \prime}+\omega^{2} y=0$ 的解, 并写出该方程的通解.
3. 验证 $y_{1}=\mathrm{e}^{x^{2}}$ 及 $y_{2}=x \mathrm{e}^{x^{2}}$ 都是方程 $y^{\prime \prime}-4 x y^{\prime}+\left(4 x^{2}-2\right) y=0$ 的解,并写出该方程的 通解。
4. 验证:

（1） $y=C_{1} \mathrm{e}^{s}+C_{2} \mathrm{e}^{2 x}+\frac{1}{12} \mathrm{e}^{s x}\left(C_{1} 、 C_{2}\right.$ 是任意常数 $)$ 是方程 $y^{N}-3 y^{\prime}+2 y=\mathrm{e}^{\mathrm{s} x}$ 的通解;

(2) $y=C_{1} \cos 3 x+C_{2} \sin 3 x+\frac{1}{32}(4 x \cos x+\sin x)\left(C_{1} 、 C_{2}\right.$ 是任意常数)是方程 $y^{\prime \prime}+9 y$ $=x \cos x$ 的通解;

(3) $y=C_{1} x^{2}+C_{2} x^{2} \ln x\left(C_{1} 、 C_{2}\right.$ 是任意常数) 是方程 $x^{2} y^{\prime \prime}-3 x y^{\prime}+4 y=0$ 的通解;

(4) $y=C_{1} x^{3}+\frac{C_{2}}{x}-\frac{x^{2}}{9} \ln x\left(C_{1} 、 C_{2}\right.$ 是任意常数) 是方程 $x^{2} y^{\prime \prime}-3 x y^{\prime}-5 y=x^{2} \ln x$ 的 通解 ;

(5) $y=\frac{1}{x}\left(C_{1} \mathrm{e}^{x}+C_{2} \mathrm{e}^{-x}\right)+\frac{\mathrm{e}^{\prime \prime}}{2}\left(C_{1} 、 C_{2}\right.$ 是任意常数 $)$ 是方程 $x y^{\prime \prime}+2 y^{\prime}-x y=\mathrm{e}^{x}$ 的通 解.

(6) $y=C_{1} \mathrm{e}^{x}+C_{2} \mathrm{e}^{-x}+C_{3} \cos x+C_{4} \sin x-x^{2}\left(C_{1} 、 C_{2} 、 C_{3} 、 C_{4}\right.$ 是任意常数) 是方程 $y^{(4)}-y=x^{2}$ 的通解.

-5. 已知 $y_{1}(x)=\mathrm{e}^{4}$ 是齐次线性方程

$$
(2 x-1) y^{\prime \prime}-(2 x+1) y^{\prime}+2 y=0
$$

的一个解, 求此方程的通解.

-6. 已知 $y_{1}(x)=x$ 是齐次线性方程 $x^{2} y^{\prime \prime}-2 x y^{\prime}+2 y=0$ 的一个解,求非齐次线性方程 $x^{2} y^{\prime \prime}-2 x y^{\prime}+2 y=2 x^{3}$ 的通解.

7. 已知齐次线性方程 $y^{\prime \prime}+y=0$ 的通解为 $Y(x)=C_{1} \cos x+C_{2} \sin x$, 求非齐次线性方 程 $y^{\prime \prime}+y=\sec x$ 的通解.
8. 已知齐次线性方程 $x^{2} y^{\prime \prime}-x y^{\prime}+y=0$ 的通解为 $Y(x)=C_{1} x+C_{2} x \ln |x|$, 求非齐次 线性方程 $x^{2} y^{\prime \prime}-x y^{\prime}+y=x$ 的通解.

## 第七节 常系数齐次线性微分方程

先讨论二阶常系数齐次线性微分方程的解法, 再把二阶方程的解法推广到 $n$ 阶方程.

在二阶齐次线性微分方程

$$
y^{\prime \prime}+P(x) y^{\prime}+Q(x) y=0
$$

中, 如果 $y^{\prime} 、 y$ 的系数 $P(x) 、 Q(x)$ 均为常数, 即 (1) 式成为

$$
y^{\prime \prime}+p y^{\prime}+q y=0 \text {. }
$$

其中 $p 、 q$ 是常数, 则称 (2) 为三阶常系数齐次线性微分方程，如果 $p 、 q$ 不全为 常数, 称 $(1)$ 为三阶变系数齐次线性微分方程.

由上节讨论可知,要找微分方程(2)的通解, 可以先求出它的两个解 $y_{1}, y_{2}$, 如果 $\frac{y_{2}}{y_{1}}$ 邦常数, 即 $y_{1}$ 与 $y_{2}$ 线性无关, 那么 $y=C_{1} y_{1}+C_{2} y_{2}$ 就是方程 (2) 的通 解.

当 $r$ 为常数时,指数函数 $y=\mathrm{e}^{r x}$ 和它的备阶导数都只相差一个常数因子. 由于指数函数有这个特点, 因此我们用 $y=\mathrm{e}^{r x}$ 来尝试, 看能否选取适当的常数 $r$, 使 $y=\mathrm{e}^{r x}$ 满足方程 (2).

将 $y=\mathrm{e}^{r: r}$ 求导 (1), 得到

$$
y^{\prime}=r \mathrm{e}^{r x}, \quad y^{\prime \prime}=r^{2} \mathrm{e}^{r x}
$$

把 $y 、 y^{\prime}$ 和 $y^{\prime \prime}$ 代入方程 (2), 得

$$
\left(r^{2}+p r+q\right) \mathrm{e}^{r x}=0 .
$$

由于 $\mathrm{e}^{r x} \neq 0$, 所以

$$
r^{2}+p r+q=0 .
$$

由此可见, 只要 $r$ 满足代数方程 (3), 函数 $y=\mathrm{e}^{r \cdot r}$ 就是微分方程 (2) 的解, 我 们把代数方程 (3) 叫做微分方程 (2) 的特征方程.

特征方程 (3) 是一个二次代数方程, 其中 $r^{2} 、 r$ 的系数及常数项恰好依次是 微分方程(2)中 $y^{\prime \prime} 、 y^{\prime}$ 及 $y$ 的系数.

$$
e^{(a t+i t) t}=c^{a t}(\cos b x+i \sin b x)
$$

两端求导。得

$$
\begin{aligned}
\frac{d}{d x} e^{(a+i b) x} & =a e^{a x r}(\cos b x+i \sin b x)+e^{a x}(-b \sin b x+i b(c) b x) \\
& =(a+i b) e^{a x}(\cos b x+i \sin b x)=(a+i b) e^{(a+i b) \cdot r} .
\end{aligned}
$$

特征方程 (3) 的两个根 $r_{1} 、 r_{2}$ 可以用公式

$$
r_{1.2}=\frac{-p \pm \sqrt{p^{2}-4 q}}{2}
$$

求出.它们有三种不同的情形:

（i）当 $p^{2}-4 q>0$ 时, $r_{1}, r_{2}$ 是两个不相等的实根：

$$
r_{1}=\frac{-p+\sqrt{p^{2}-4 q}}{2}, r_{2}=\frac{-p-\sqrt{p^{2}-4 q}}{2} ;
$$

(ii) 当 $p^{2}-4 q=0$ 时, $r_{1}, r_{2}$ 是两个相等的实根：

$$
r_{1}=r_{2}=-\frac{p}{2} ;
$$

(iii) 当 $p^{2}-4 q<0$ 时, $r_{1}, r_{2}$ 是一对共轭复根:

$$
r_{1}=\alpha+\mathrm{i} \beta, \quad r_{2}=\alpha-\mathrm{i} \beta,
$$

其中

$$
\alpha=-\frac{p}{2}, \quad \beta=\frac{\sqrt{4 q-p^{2}}}{2} .
$$

相应的, 微分方程 (2) 的通解也有三种不同的情形. 分别讨论如下:

(i) 特征方程有两个不相等的实根: $r_{1} \neq r_{2}$.

由上面的讨论知道, $y_{1}=\mathrm{e}^{r_{1} r} 、 y_{2}=\mathrm{e}^{r_{2} x}$ 是微分方程 (2) 的两个解, 并且 $\frac{y_{2}}{y_{1}}=$ $\frac{\mathrm{e}^{r_{2} x}}{\mathrm{e}^{r_{1} x^{x}}}=\mathrm{e}^{\left(r_{2}-r_{1}\right) x}$ 不是常数, 因此微分方程 (2)的通解为

$$
y=C_{1} \mathrm{e}^{r_{1}{ }^{\prime}}+C_{2} \mathrm{e}^{r_{2} \cdot r} .
$$

(ii) 特征方程有两个相等的实根: $r_{1}=r_{2}$.

这时, 只得到微分方程 (2) 的一个解

$$
y_{1}=\mathrm{e}^{r_{1} \cdot x} \text {. }
$$

为了得出微分方程 (2) 的通解, 还需求出另一个解 $y_{2}$, 并且要求 $\frac{y_{2}}{y_{1}}$ 不是常 数.

设 $\frac{y_{2}}{y_{1}}=u(x)$, 即 $y_{2}=\mathrm{e}^{r_{1} \cdot r} u(x)$. 下面来求 $u(x)$.

将 $y_{2}$ 求导, 得

$$
\begin{aligned}
& y_{2}^{\prime}=\mathrm{e}^{r_{1} x}\left(u^{\prime}+r_{1} u\right), \\
& y_{2}^{\prime \prime}=\mathrm{e}^{r_{1} x}\left(u^{\prime \prime}+2 r_{1} u^{\prime}+r_{1}^{2} u\right),
\end{aligned}
$$

将 $y_{2} 、 y_{2}^{\prime}$ 和 $y_{2}^{\prime \prime}$ 代入微分方程(2), 得

$$
\mathrm{e}^{r_{1} x^{x}}\left[\left(u^{\prime \prime}+2 r_{1} u^{\prime}+r_{1}^{2} u\right)+p\left(u^{\prime}+r_{1} u\right)+q u\right]=0 \text {, }
$$

约去 $\mathrm{e}^{r_{1},{ }^{x}}$,并以 $u^{\prime \prime} 、 u^{\prime} 、 u$ 为准合并同类项, 得

$$
u^{\prime \prime}+\left(2 r_{1}+p\right) u^{\prime}+\left(r_{1}^{2}+p r_{1}+q\right) u=0 .
$$

由于 $r_{1}$ 是特征方程 (3) 的二重根. 因此 $r_{1}^{2}+p r_{1}+q=0$, 且 $2 r_{1}+p=0$, 于 是得

$$
u^{\prime \prime}=0 .
$$

因为这里只要得到一个不为常数的解, 所以不妨选取 $u=x$, 由此得到微分方程 (2)的另一个解

$$
y_{2}=x \mathrm{e}^{r_{1} x} .
$$

从而微分方程 (2) 的通解为

即

$$
y=C_{1} \mathrm{e}^{r_{1} x}+C_{2} x \mathrm{e}^{r_{1} x},
$$

$$
y=\left(C_{1}+C_{2} x\right) \mathrm{e}^{r_{1} x} \text {. }
$$

(iii) 特征方程有一对共轭复根; $r_{1}=\alpha+\mathrm{i} \beta, r_{2}=\alpha-\mathrm{i} \beta(\beta \neq 0)$.

这时, $y_{1}=\mathrm{e}^{(\alpha+i \beta) . x}, y_{2}=\mathrm{e}^{(a-i \beta) . x}$ 是微分方程 (2) 的两个解, 但它们是复值函数 形式. 为了得出实值函数形式的解, 先利用欧拉公式 $\mathrm{e}^{\mathrm{i} \theta}=\cos \theta+\mathrm{i} \sin \theta$ 把 $y_{1} 、 y_{2}$ 改写为

$$
\begin{aligned}
& y_{1}=\mathrm{e}^{(\alpha+\mathrm{i} \beta) x}=\mathrm{e}^{\alpha r} \cdot \mathrm{e}^{\mathrm{i} \beta r}=\mathrm{e}^{a x}(\cos \beta x+\mathrm{i} \sin \beta x), \\
& y_{2}=\mathrm{e}^{(\alpha-\mathrm{i} \beta) \cdot x}=\mathrm{e}^{\alpha x} \cdot \mathrm{e}^{-\mathrm{i} \beta r}=\mathrm{e}^{a x}(\cos \beta x-\mathrm{i} \sin \beta x) .
\end{aligned}
$$

由于复值函数 $y_{1}$ 与 $y_{2}$ 之间成共轭关系, 因此, 取它们的和除以 2 就得到它们的 实部; 取它们的差除以 $2 \mathrm{i}$ 就得到它们的虚部. 由于方程 (2) 的解符合叠加原理, 所以实值函数

$$
\begin{aligned}
& \bar{y}_{1}=\frac{1}{2}\left(y_{1}+y_{2}\right)=\mathrm{e}^{a x} \cos \beta x, \\
& \bar{y}_{2}=\frac{1}{2 \mathrm{i}}\left(y_{1}-y_{2}\right)=\mathrm{e}^{a x} \sin \beta x
\end{aligned}
$$

还是微分方程 (2) 的解, 且 $\frac{\bar{y}_{1}}{\bar{y}_{2}}=\frac{\mathrm{e}^{a x} \cos \beta x}{\mathrm{e}^{a .} \sin \beta x}=\cot \beta x$ 不是常数, 所以微分方程 (2) 的 通解为

$$
y=\mathrm{e}^{\alpha x}\left(C_{1} \cos \beta x+C_{2} \sin \beta x\right) .
$$

综上所述, 求二阶常系数齐次线性微分方程

$$
y^{\prime \prime}+p y^{\prime}+q y=0
$$

的通解的步骤姆下:

第一步 写出微分方程 (2) 的特征方程

$$
r^{2}+p r+q=0 \text {. }
$$

第二步 求出特征方程 (3) 的两个根 $r_{1}, r_{2}$.

第三步 根据特征方程 (3) 的两个根的不同情形, 按照下列表格写出微分方 程 (2) 的通解:

| 特征方程 $r^{2}+p r+q=0$ 的两个根 $r_{1}, r_{2}$ | 微分方程 $y^{\prime \prime}+p y^{\prime}+q y=0$ 的通解 |
| :--- | :--- |
| 两个不相等的实根 $r_{1}, r_{2}$ | $y=C_{1} \mathrm{e}^{r_{1} x}+C_{2} \mathrm{e}^{r_{2}} \cdot x$ |
| 两个相等的实根 $r_{1}=r_{2}$ | $y=\left(C_{1}+C_{2} x\right) \mathrm{e}^{r_{1} x}$ |
| 一对共轭复根 $r_{1,2}=\alpha \pm \mathrm{i} \beta$ | $y=\mathrm{e}^{a x}\left(C_{1} \cos \beta x+C_{2} \sin \beta x\right)$ |

例 1 求微分方程 $y^{\prime \prime}-2 y^{\prime}-3 y=0$ 的通解.

解 所给微分方程的特征方程为

$$
r^{2}-2 r-3=0,
$$

其根 $r_{1}=-1, r_{2}=3$ 是两个不相等的实根, 因此所求通解为

$$
y=C_{1} \mathrm{e}^{-x}+C_{2} \mathrm{e}^{3 x} .
$$

例 2 求方程 $\frac{\mathrm{d}^{2} s}{\mathrm{~d} t^{2}}+2 \frac{\mathrm{d} s}{\mathrm{~d} t}+s=0$ 满足初始条件 $\left.s\right|_{t=0}=\left.4 、 s^{\prime}\right|_{t=0}=-2$ 的特 解.

解 所给方程的特征方程为

$$
r^{2}+2 r+1=0,
$$

其根 $r_{1}=r_{2}=-1$ 是两个相等的实根, 因此所求微分方程的通解为

$$
s=\left(C_{1}+C_{2} t\right) \mathrm{e}^{-t} .
$$

将条件 $\left.s\right|_{t=0}=4$ 代入通解, 得 $C_{1}=4$, 从而

$$
s=\left(4+C_{2} t\right) \mathrm{e}^{-t} \text {. }
$$

将上式对 $t$ 求导, 得

$$
s^{\prime}=\left(C_{2}-4-C_{2} t\right) \mathrm{e}^{-t} .
$$

再把条件 $\left.s^{\prime}\right|_{t=0}=-2$ 代入上式, 得 $C_{2}=2$.于是所求特解为

$$
s=(4+2 t) \mathrm{e}^{-t} \text {. }
$$

例 3 求微分方程 $y^{\prime \prime}-2 y^{\prime}+5 y=0$ 的通解.

解 所给方程的特征方程为

$$
r^{2}-2 r+5=0,
$$

其根 $r_{1,2}=1 \pm 2 \mathrm{i}$ 为一对共轮复根. 因此所求通解为

$$
y=\mathrm{e}^{x}\left(C_{1} \cos 2 x+C_{2} \sin 2 x\right) \text {. }
$$

例 4 在第六节例 1 中, 设物体只受弹性恢复力 $f$ 的作用, 且在初瞬 $t=0$ 时的位置为 $x=x_{0}$, 初始速度为 $\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=0}=v_{0}$. 求反咉物体运动规律的函数 $x=$ $x(t)$.

解 由于不计阻力 $R$, 即假设 $-\mu \frac{\mathrm{d} x}{\mathrm{~d} t}=0$, 所以第六节中的方程 (1) 成为

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+k^{2} x=0,
$$

方程 (4)叫做无阻尼息由振动的微分方程.

反映物体运动规律的函数 $x=x(t)$ 是满足微分方程 (4) 及初始条件

$$
\left.x\right|_{1=0}=x_{0},\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=11}=v_{0}
$$

的特解.

方程 (4) 的特征方程为 $r^{2}+k^{2}=0$, 其根 $r= \pm i k$ 是一对共轭复根, 所以方 程 (4) 的通解为

$$
x=C_{1} \cos k t+C_{2} \sin k t .
$$

应用初始条件, 定出 $C_{1}=x_{0}, C_{2}=\frac{v_{0}}{k}$. 因此, 所求的特解为

$$
x=x_{0} \cos k t+\frac{v_{11}}{k} \sin k t .
$$

为了便于说明特解所反映的振动现象, 我们令

$$
x_{0}=A \sin \varphi, \quad \frac{v_{0}}{k}=A \cos \varphi \quad(0 \leqslant \varphi<2 \pi),
$$

于是 (5) 式成为

$$
x=A \sin (k t+\varphi),
$$

其中

$$
A=\sqrt{x_{0}^{2}+\frac{v_{0}^{2}}{k^{2}}}, \quad \tan \varphi=\frac{k x_{0}}{v_{0}} .
$$

函数 (6) 的图形如图 7-10 所示(图中假定 $x_{0}>0, v_{0}>0$ ).

函数 (6) 所反映的运动就是䈍谐振动. 这个振动的振幅为 $A$, 初相为 $\varphi$, 周期 为 $T=\frac{2 \pi}{k}$, 角频率为 $k$. 由于 $k=\sqrt{\frac{c}{m}}$ (元第六节例 1), 它与初始条件无关, 而 完全由振动系统 (在本例中就是弹簧和物体所组成的系统) 本身所确定. 因此, $k$ 又叫做系统的固有频率. 固有频率是反映振动系统特性的一个重要参数.

例 5 在第六节例 1 中, 设物体受弹签的恢复力 $f$ 和阻力 $R$ 的作用, 且在初 瞬 $t=0$ 时的位置 $x=x_{0}$, 初始速度 $\left.\frac{\mathrm{d} x}{\mathrm{~d} t}\right|_{t=0}=v_{0}$, 求反映物体运动规律的函数 $x=x(t)$.

解 这就是要找满足有阻尼的自由振动方程

及初始条件

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+2 n \frac{\mathrm{d} x}{\mathrm{~d} t}+k^{2} x=0
$$

的特解.

方程 (7) 的特征方程为 $r^{2}+2 n r+k^{2}=0$, 其根为

$$
r=\frac{-2 n \pm \sqrt{4 n^{2}-4 k^{2}}}{2}=-n \pm \sqrt{n^{2}-k^{2}} \text {. }
$$

以下按 $n<k, n>k$ 及 $n=k$ 三种不同情形分别进行讨论.

（i）小阻尼情形: $n<k$.

特征方程的根 $r=-n \pm \mathrm{i} \omega\left(\omega=\sqrt{k^{2}-n^{2}}\right)$ 是一对共轭复根, 所以方程 (7) 的通解为

$$
x=\mathrm{e}^{-m t}\left(C_{1} \cos \omega t+C_{2} \sin \omega t\right) .
$$

应用初始条件定出 $C_{1}=x_{11}, C_{2}=\frac{v_{01}+n x_{0}}{\omega}$, 因此所求特解为

$$
x=\mathrm{e}^{-n t}\left(x_{0} \cos \omega t+\frac{v_{0}+n x_{0}}{\omega} \sin \omega t\right) .
$$

如例 4 中所作的那样, 令

$$
x_{0}=A \sin \varphi, \quad \frac{v_{11}+n x_{11}}{\omega}=A \cos \varphi(0 \leqslant \varphi<2 \pi),
$$

那么(8)式又可写成

$$
x=A \mathrm{e}^{-n t} \sin (\omega t+\varphi),
$$

其中

$$
\omega=\sqrt{k^{2}-n^{2}}, A=\sqrt{x_{0}^{2}+\frac{\left(v_{0}+n x_{0}\right)^{2}}{\omega^{2}}}, \quad \tan \varphi=\frac{x_{11} \omega}{v_{0}+n x_{11}} .
$$

从 $(10)$ 式看出,物体的运动是周期 $T=\frac{2 \pi}{\omega}$ 的振动. 但与简谐振动不同, 它的 振幅 $A \mathrm{e}^{-n t}$ 随时间 $t$ 的增大而逐渐减小. 因此, 物体随时间 $t$ 的增大而趋于平衡 位置.

函数 (10) 的图形如图 7-11 所示(图中假定 $x_{\mathfrak{n}}=0, v_{0}>0$ ).

（ii）大阻尼情形: $n>k$.

特征方程的根 $r_{1}=-n+\sqrt{n^{2}-k^{2}}, r_{2}=-n-\sqrt{n^{2}-k^{2}}$. 是两个不相等的 负实根, 所以方程 (7) 的通解为

$$
x=C_{1} \mathrm{e}^{-\left(n \cdot \sqrt{n^{2}-k^{2}}\right) t}+C_{2} \mathrm{e}^{-\left(n+\sqrt{n^{2}-k^{2}}\right) t},
$$

其中任意常数 $C_{1} 、 C_{2}$ 可以由初始条件来确定.

从(11)式看出, 使 $x=0$ 的 $t$ 值最多只有一个, 即物体最多越过平衡位置一 次, 因此物体已不再有振动现象. 又当 $t \rightarrow+\infty$ 时, $x \rightarrow 0$. 因此,物体随时间 $t$ 的 增大而趋于平衡位置.

函数 (11) 的图形如图 7-12 所示 (图中假定 $x_{0}>0, v_{0}>0$ ).

（iii）临界阻尼情形: $n=k$.

特征方程的根 $r_{1}=r_{2}=-n$ 是两个相等的实根, 所以方程 (7) 的通解为

$$
x=\mathrm{e}^{-m}\left(C_{1}+C_{2} t\right) \text {, }
$$

其中任意常数 $C_{1}$ 及 $C_{3}$ 可由初始条件来确定: 由上式可看出, 在临界阻尼情形 使 $x=0$ 的 $t$ 值也最多只有一个, 因此物体也不再有振动现象. 又由于

$$
\lim _{n \rightarrow+\infty} t \mathrm{e}^{-n t}=\lim _{t \rightarrow+\infty} \frac{t}{\mathrm{e}^{n t}}=\lim _{t \rightarrow+\infty} \frac{1}{n \mathrm{e}^{n t}}=0,
$$

从而可以看出, 当 $i \rightarrow+\infty$ 时, $x \rightarrow 0$. 因此, 在临界阻尼情形, 物体也随时间 $t$ 的 增大而趋于平衡位置.

上面讨论二阶常系数齐次线性微分方程所用的方法以及方程的通解的形 式, 可推广到 $n$ 阶常系数齐次线性微分方程上去, 对此我们不再详细讨论, 只简 单地叙述于下:

$n$ 阶常系数卉次线性微分方程的一般形式是

$$
y^{(n)}+p_{1} y^{(n-1)}+p_{2} y^{(n-2)}+\cdots+p_{n-1} y^{\prime}+p_{n} y=0,
$$

其中 $p_{1}, p_{2}, \cdots, p_{n}, p_{n}$ 都是常数.

有时我们用记昂 D (叫做微分算子) 表示对 $x$ 求导的运算 $\frac{\mathrm{d}}{\mathrm{d} x}$, 把 $\frac{\mathrm{d} y}{\mathrm{~d} x}$ 记作 $\mathrm{D} y$, 把 $\frac{\mathrm{d}^{n} y}{\mathrm{~d} x^{n}}$ 记作 $\mathrm{D}^{n} y$,并把方程 (12)记作

$$
\begin{aligned}
& \left(\mathrm{D}^{n}+p_{1} \mathrm{D}^{n-1}+\cdots+p_{n-1} \mathrm{D}+p_{n}\right) y=0 . \\
& \text { 记 } \quad L(\mathrm{D})=\mathrm{D}^{n}+p_{1} \mathrm{D}^{n-1}+\cdots+p_{n-1} \mathrm{D}+p_{n},
\end{aligned}
$$

$L(D)$ 叫做微分算子 $\mathrm{D}$ 的 $n$ 次多项式. 于是方程 (13) 可记作

$$
L(D) y=0 \text {. }
$$

如同讨论二阶常系数齐次线性微分方程那样, 令 $y=\mathrm{e}^{r . x}$. 由于 $\mathrm{De}^{r . x}=r \mathrm{e}^{r x}$, $\cdots, \mathrm{D}^{n} \mathrm{e}^{r x}=r^{n} \mathrm{e}^{r x}$, 故 $L(\mathrm{D}) \mathrm{e}^{r x}=L(r) \mathrm{e}^{r x}$. 因此把 $y=\mathrm{e}^{r x}$ 代入方程 (13), 得

$$
L(r) \mathrm{e}^{r x}=0 \text {. }
$$

由此可见,如果选取 $r$ 是 $n$ 次代数方程

$$
L(r)=0 \text { 即 } r^{n}+p_{1} r^{n-1}+p_{2} r^{n-2}+\cdots+p_{n-1} r+p_{n}=0
$$

| 特征方程的根 | 微分方程通解中的对应项 |
| :---: | :---: |
| 单实根 $r$ | 给出一项: $C \mathrm{e}^{r \cdot x}$ |
| 一对单变根 | 给出两项: $\mathrm{e}^{\mathrm{ar}}\left(C_{1} \cos \beta x+C_{2} \sin \beta x\right)$ |
| $r_{1,2}=\alpha \pm \mathrm{i} \beta$ | $\cdot$ |
| $k$ 重实根 $r$ | 给出 $k$ 项: $\mathrm{e}^{r s}\left(C_{1}+C_{2} x+\cdots+C_{k} x^{k-1}\right)$ |
| 一对 $k$ 重妏根 | 给出 $2 k$ 项: $\mathrm{e}^{a r}\left[\left(C_{1}+C_{2} x+\cdots+C_{k} \cdot x^{4-1}\right) \cos \beta x+\left(D_{1}+D_{2} x+\cdots+\right.\right.$ |
| $r_{1.2}=\alpha \pm i \beta$ | $\left.\left.D_{k} x^{k-1}\right) \sin \beta x\right]$ |

的根, 那么作出的函数 $y=\mathrm{e}^{r . r}$ 就是方程 (13) 的一个解.

方程 (14) 叫做方程 (13) 的特征方程.

根据特征方程的根, 可以写出其对应的微分方程的解如下:

从代数学知道, $n$ 次代数方程有 $n$ 个根 (重根按重数计算). 而特征方程的每 一个根都对应着通解中的一项, 且每项各含一个任意常数. 这样就得到 $n$ 阶常 系数齐次线性微分方程的通解

$$
y=C_{1} y_{1}+C_{2} y_{2}+\cdots+C_{n} y_{n} .
$$

例 6 求方程 $y^{(4)}-2 y^{\prime \prime \prime}+5 y^{\prime \prime}=0$ 的通解.

解 这里的特征方程为

即

$$
\begin{gathered}
r^{4}-2 r^{3}+5 r^{2}=0, \\
r^{2}\left(r^{2}-2 r+5\right)=0 .
\end{gathered}
$$

它的根是 $r_{1}=r_{2}=0$ 和 $r_{3,4}=1 \pm 2 \mathrm{i}$.

因此所给微分方程的通解为

$$
y=C_{1}+C_{2} x+\mathrm{e}^{x}\left(C_{3} \cos 2 x+C_{4} \sin 2 x\right) .
$$

例 7 求方程 $\frac{d^{4} w}{d x^{4}}+\beta^{4} w=0$ 的通解, 其中 $\beta>0$. 解 这里的特征方程为

$$
r^{4}+\beta^{4}=0
$$

由于

$$
\begin{aligned}
r^{4}+\beta^{4} & =r^{4}+2 r^{2} \beta^{2}+\beta^{4}-2 r^{2} \beta^{2}=\left(r^{2}+\beta^{2}\right)^{2}-2 r^{2} \beta^{2} \\
& =\left(r^{2}-\sqrt{2} \beta r+\beta^{2}\right)\left(r^{2}+\sqrt{2} \beta r+\beta^{2}\right),
\end{aligned}
$$

所以特征方程可以写为

$$
\left(r^{2}-\sqrt{2} \beta r+\beta^{2}\right)\left(r^{2}+\sqrt{2} \beta r+\beta^{2}\right)=0 .
$$

它的根为 $r_{1,2}=\frac{\beta}{\sqrt{2}}(1 \pm i), r_{3,4}=-\frac{\beta}{\sqrt{2}}(1 \pm i)$, 因此所给方程的通解为

$$
w=\mathrm{e}^{\frac{\beta}{\sqrt{2}} x}\left(C_{1} \cos \frac{\beta}{\sqrt{2}} x+C_{2} \sin \frac{\beta}{\sqrt{2}} x\right)+\mathrm{e}^{-\frac{\beta}{\sqrt{2}} x}\left(C_{3} \cos \frac{\beta}{\sqrt{2}} x+C_{4} \sin \frac{\beta}{\sqrt{2}} x\right) .
$$

## 习 题 7-7

1. 求下列微分方程的通解:
(1) $y^{\prime \prime}+y^{\prime}-2 y=0$;
(2) $y^{\prime \prime}-4 y^{\prime}=0$;
(3) $y^{\prime \prime}+y=0$;
(4) $y^{\prime \prime}+6 y^{\prime}+13 y=0$;
(5) $4 \frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}-20 \frac{\mathrm{d} x}{\mathrm{~d} t}+25 x=0$;
(6) $y^{\prime \prime}-4 y^{\prime}+5 y=0$;
(7) $y^{(4)}-y=0$;
(8) $y^{(4)}+2 y^{\prime \prime}+y=0$;
(9) $y^{(4)}-2 y^{\prime \prime \prime}+y^{\prime \prime}=0$;
(10) $y^{(4)}+5 y^{\prime \prime}-36 y=0$.
2. 求下列微分方程满足所给初始条件的特解:
(1) $y^{\prime \prime}-4 y^{\prime}+3 y=0,\left.y\right|_{x=0}=6,\left.y^{\prime}\right|_{x:=11}=10$;
(2) $4 y^{\prime \prime}+4 y^{\prime}+y=0,\left.y\right|_{x=0}=2,\left.y^{\prime}\right|_{r=0}=0$;
(3) $y^{\prime \prime}-3 y^{\prime}-4 y=0,\left.y\right|_{x=0}=0,\left.y^{\prime}\right|_{x=11}=-5$;
(4) $y^{\prime \prime}+4 y^{\prime}+29 y=0,\left.y\right|_{, r<11}=0,\left.y^{\prime}\right|_{\text {, }-11}=15$;
(5) $y^{\prime \prime}+25 y=0,\left.y\right|_{. r=11}=2,\left.y^{\prime}\right|_{\text {.ral1 }}=5$;
(6) $y^{\prime \prime}-4 y^{\prime}+13 y=0,\left.y\right|_{, x=11}=0,\left.y^{\prime}\right|_{x=0}=3$.
3. 一个单位质的质点在数轴上运动, 开始时质点在原点 $O$ 处且速度为 $v_{0}$, 在运动过程中, 它受到一个力的作用, 这个力的大 小与质点到原点的距离成正比(比例系数 $k_{1}>0$ ) 而方向与初速度 一致. 又介质的阻力与速度成正比 (比例系数 $k_{2}>0$ ). 求反映这质 点的运动规律的函数.
4. 在图 7-13 所示的电路中先将开关 $S$ 拨问 $A$, 达到稳定状 态后再将开关 $\mathrm{S}$ 拨向 $B$, 求电压 $u_{c}(t)$ 及电流 $i(t)$. 已知 $E=$

## 第八节 常系数非齐次线性微分方程

本节着重讨论二阶常系数非齐次线性微分方程的解法,并对 $n$ 阶方程的解 法作必要的说明.

二阶常系数非齐次线性微分方程的一般形式是

$$
y^{\prime \prime}+p y^{\prime}+q y=f(x),
$$

其中 $p 、 q$ 是常数.

由第六节定理 3 可知, 求二阶常系数非齐次线性微分方程的通解, 归结为求 对应的齐次方程

$$
y^{\prime \prime}+p y^{\prime}+q y=0
$$

的通解和非齐次方程 (1)本身的一个特解. 由于二阶常系数齐次线性微分方程的 通解的求法已在第七节得到解决, 所以这里只需讨论求二阶常系数非齐次线性 微分方程的一个特解 $y^{*}$ 的方法.

本节只介绍当方程 (1) 中的 $f(x)$ 取两种常见形式时求 $y^{*}$ 的方法. 这种方 法的特点是不用积分就可求出 $y^{*}$ 来, 它叫做待定系数法. $f(x)$ 的两种形式是

(1) $f(x)=P_{m}(x) \mathrm{e}^{\lambda x}$, 其中 $\lambda$ 是常数, $P_{m}(x)$ 是 $x$ 的一个 $m$ 次多项式:

$$
P_{m}(x)=a_{0} x^{m}+a_{1} x^{m-1}+\cdots+a_{m-1} x+a_{m} ;
$$

(2) $f(x)=\mathrm{e}^{\lambda x}\left[P_{l}(x) \cos \omega x+P_{n}(x) \sin \omega x\right]$, 其中 $\lambda 、 \omega$ 是常数, $P_{l}(x)$ 、 $P_{n}(x)$ 分别是 $x$ 的 $l$ 次、 $n$ 次多项式, 且有一个可为零.

下面分别介绍 $f(x)$ 为上述两种形式时 $y^{*}$ 的求法.

$$
\text { 一、 } f(x)=\mathrm{e}^{\lambda x} P_{m}(x) \text { 型 }
$$

我们知道,方程 (1) 的特解 $y^{*}$ 是使 (1) 成为恒等式的函数: 怎样的函数能使 (1) 成为恒等式呢? 因为 (1) 式右端 $f(x)$ 是多项式 $P_{m}(x)$ 与指数函数 $\mathrm{e}^{\lambda x}$ 的乘 积, 而多项式与指数函数乘积的导数仍然是多项式与指数函数的乘积, 因此, 我 们推测 $y^{*}=Q(x) \mathrm{e}^{\lambda x}$ (其中 $Q(x)$ 是某个多项式)可能是方程 (1) 的特解. 把 $y^{*} 、 y^{*}$ '及 $y^{*}$ "代入方程 (1), 然后考虑能否选取适当的多项式 $Q(x)$, 使 $y^{*}=$ $Q(x) \mathrm{e}^{\lambda x}$ 满足方程 (1). 为此, 将

$$
\begin{aligned}
& y^{\prime \prime}=Q(x) \mathrm{e}^{\lambda x}, \\
& y^{\prime \prime}=\mathrm{e}^{\lambda x}\left[\lambda Q(x)+Q^{\prime}(x)\right], \\
& y^{\prime \prime \prime}=\mathrm{e}^{\lambda x}\left[\lambda^{2} Q(x)+2 \lambda Q^{\prime}(x)+Q^{\prime \prime}(x)\right]
\end{aligned}
$$

代入方程 (1) 并消去 $\mathrm{e}^{\lambda x}$, 得

$$
Q^{\prime \prime}(x)+(2 \lambda+p) Q^{\prime}(x)+\left(\lambda^{2}+p \lambda+q\right) Q(x)=P_{m}(x) .
$$

(i) 如果 $\lambda$ 不是 (2) 式的特征方程 $r^{2}+p r+q=0$ 的根, 即 $\lambda^{2}+p \lambda+q \neq 0$, 由于 $P_{m}(x)$ 是一个 $m$ 次多项式,要使 (3) 的两端恒等, 那么可令 $Q(x)$ 为另一 个 $m$ 次多项式 $Q_{m}(x)$ :

$$
Q_{m}(x)=b_{0} x^{m+}+b_{1} x^{m-1}+\cdots+b_{m-1} x+b_{m},
$$

代入 (3) 式, 比较等式两端 $x$ 同次帛的系数, 就得到以 $b_{0}, b_{1}, \cdots, b_{m}$ 作为未知数 的 $m+1$ 个方程的联立方程组. 从而可以定出这些 $b_{i}(i=0,1, \cdots, m)$, 并得到所 求的特解 $y^{*}=Q_{m}(x) \mathrm{e}^{i x}$.

(ii) 如果 $\lambda$ 是特征方程 $r^{2}+p r+q=0$ 的单根, 即 $\lambda^{2}+p \lambda+q=0$, 但 $2 \lambda+p$ $\neq 0$, 要使 (3) 的两端恒等, 那么 $Q^{\prime}(x)$ 必须是 $m$ 次多项式. 此时可令

$$
Q(x)=x Q_{m}(x),
$$

并且可用同样的方法来确定 $Q_{m}(x)$ 的系数 $b_{i} \quad(i=0,1,2, \cdots, m)$.

(iii) 如果 $\lambda$ 是特征方程 $r^{2}+p r+q=0$ 的重根, 即 $\lambda^{2}+p \lambda+q=0$, 且 $2 \lambda+$ $p=0$, 要使 (3) 的两端恒等, 那么 $Q^{\prime \prime}(x)$ 必须是 $m$ 次多项式. 此时可令

$$
Q(x)=x^{2} Q_{m}(x) \text {, }
$$

并用同样的方法来确定 $Q_{m}(x)$ 中的系数.

综上所述, 我们有如下结论:

如果 $f(x)=P_{m}(x) \mathrm{e}^{\lambda x}$, 则二阶常系数非齐次线性微分方程 (1) 具有形如

$$
y^{*}=x^{k} Q_{m}(x) \mathrm{e}^{\lambda r}
$$

的特解, 其中 $Q_{m}(x)$ 是与 $P_{m}(x)$ 同次 ( $m$ 次) 的多项式, 而 $k$ 按 $\lambda$ 不是特征方程 的根、是特征方程的单根或是特征方程的重根依次取为 $0 、 1$ 或 2 .

上述结论可推广到 $n$ 阶常系数非齐次线性微分方程,但要注意(4)式中的 $k$ 是特征方程含根 $\lambda$ 的重复次数(即若 $\lambda$ 不是特征方程的根,则 $k$ 取为 0 ; 若 $\lambda$ 是 特征方程的 $s$ 重根, 则 $k$ 取为 $s$ ).

例 1 求微分方程 $y^{\prime \prime}-2 y^{\prime}-3 y=3 x+1$ 的一个特解.

解 这是二阶常系数非齐次线性微分方程, 且函数 $f(x)$ 是 $P_{m}(x) \mathrm{e}^{2 x}$ 型 (其 中 $\left.P_{m}(x)=3 x+1, \lambda=0\right)$.

与所给方程对应的齐次方程为

$$
y^{\prime \prime}-2 y^{\prime}-3 y=0 \text {, }
$$

它的特征方程为

$$
r^{2}-2 r-3=0 \text {. }
$$

由于这里 $\lambda=0$ 不是特征方程的根, 所以应设特解为

$$
y^{\cdot}=b_{0} x+b_{1} \text {. }
$$

把它代入所给方程,得

$$
-3 b_{0} x-2 b_{0}-3 b_{1}=3 x+1,
$$

比较两端 $x$ 同次劳的系数,得

$$
\left\{\begin{array}{l}
-3 b_{0}=3, \\
-2 b_{0}-3 b_{1}=1
\end{array}\right.
$$

由此求得 $b_{0}=-1, b_{1}=\frac{1}{3}$. 于是求得一个特解为

$$
y^{*}=-x+\frac{1}{3} \text {. }
$$

例 2 求微分方程 $y^{\prime \prime}-5 y^{\prime}+6 y=x \mathrm{e}^{2 x}$ 的通解.

解 所给方程也是二阶常系数非齐次线性微分方程, 且 $f(x)$ 呈 $P_{m}(x) \mathrm{e}^{\lambda x}$ 型(其中 $P_{m}(x)=x, \lambda=2$ ).

与所给方程对应的齐次方程为

$$
y^{\prime \prime}-5 y^{\prime}+6 y=0,
$$

它的特征方程

$$
r^{2}-5 r+6=0
$$

有两个实根 $r_{1}=2, r_{2}=3$. 于是与所给方程对应的齐次方程的通解为

$$
Y=C_{1} \mathrm{e}^{2 x}+C_{2} \mathrm{e}^{3 x} .
$$

由于 $\lambda=2$ 是特征方程的单根, 所以应设 $y^{*}$ 为

$$
y^{*}=x\left(b_{0} x+b_{1}\right) \mathrm{e}^{2 x} \text {. }
$$

把它代入所给方程,得

$$
-2 b_{0} x+2 b_{0}-b_{1}=x .
$$

比较等式两端同次幂的系数,得

$$
\left\{\begin{array}{l}
-2 b_{0}=1 \\
2 b_{0}-b_{1}=0 .
\end{array}\right.
$$

解得 $b_{0}=-\frac{1}{2}, b_{1}=-1$. 因此求得一个特解为

$$
y^{*}=x\left(-\frac{1}{2} x-1\right) \mathrm{e}^{2 x} .
$$

从而所求的通解为

$$
\begin{gathered}
y=C_{1} \mathrm{e}^{2 x}+C_{2} \mathrm{e}^{3 x}-\frac{1}{2}\left(x^{2}+2 x\right) \mathrm{e}^{2 x} . \\
=、 f(x)=\mathrm{e}^{\lambda x}\left[P_{l}(x) \cos \omega x+P_{n}(x) \sin \omega x\right] \text { 型 }
\end{gathered}
$$

应用欧拉公式

$$
\cos \theta=\frac{1}{2}\left(\mathrm{e}^{\mathrm{i} \theta}+\mathrm{e}^{-\mathrm{i} \theta}\right), \sin \theta=\frac{1}{2 \mathrm{i}}(\cos \theta-\mathrm{i} \sin \theta),
$$

把 $f(x)$ 表成复变指数函数的形式,有

$$
\begin{aligned}
f(x) & =\mathrm{e}^{\lambda x}\left[P_{1} \cos \omega x+P_{n} \sin \omega x\right] \\
& =\mathrm{e}^{\lambda x}\left[P_{l} \frac{\mathrm{e}^{\mathrm{i} \omega r}+\mathrm{e}^{-\mathrm{i} \omega x}}{2}+P_{n} \frac{\mathrm{e}^{\mathrm{i} \omega r}-\mathrm{e}^{-\mathrm{i} \omega x}}{2 \mathrm{i}}\right] \\
& =\left(\frac{P_{l}}{2}+\frac{P_{n}}{2 \mathrm{i}}\right) \mathrm{e}^{(\lambda+\mathrm{i} \omega) x}+\left(\frac{P_{l}}{2}-\frac{P_{n}}{2 \mathrm{i}}\right) \mathrm{e}^{(\lambda-\mathrm{i} \omega) x} \\
& =P(x) \mathrm{e}^{(\lambda+\mathrm{i} \omega) \cdot x}+\bar{P}(x) \mathrm{e}^{(\lambda-\mathrm{i} \omega) x} \\
P(x) & =\frac{P_{l}}{2}+\frac{P_{n}}{2 \mathrm{i}}=\frac{P_{l}}{2}-\frac{P_{n}}{2} \mathrm{i}, \\
\bar{P}(x) & =\frac{P_{l}}{2}-\frac{P_{n}}{2 \mathrm{i}}=\frac{P_{l}}{2}+\frac{P_{n}}{2} \mathrm{i}
\end{aligned}
$$

其中

是互成共轮的 $m$ 次多项式 (即它们对应项的系数是共轭复数), 而 $m=$ $\max \{l, n\}$.

应用上一目的结果, 对于 $f(x)$ 中的第一项 $P(x) \mathrm{e}^{(\lambda+i \omega) . r}$, 可求出一个 $m$ 次 多项式 $Q_{m}(x)$, 使得 $y_{\mathrm{i}}{ }^{\prime}=x^{k} Q_{m} \mathrm{e}^{(\lambda+\mathrm{i} \omega) \cdot x}$ 为方程

$$
y^{\prime \prime}+p y^{\prime}+q y=P(x) \mathrm{e}^{(\lambda+i \omega) x}
$$

的特解, 其中 $k$ 按 $\lambda+\mathrm{i} \omega$ 不是特征方程的根或是特征方程的单根依次取 0 或 1 . 由于 $f(x)$ 的第二项 $\bar{P}(x) \mathrm{e}^{(\lambda-i \omega) x}$ 与第一项 $P(x) \cdot \mathrm{e}^{(\lambda+i \omega) x}$ 成共轭, 所以与 $y_{i}$ 成 共轭的函数 $y_{2}^{*}=x^{k} \bar{Q}_{m} \mathrm{e}^{(\lambda-\mathrm{i} \omega) \cdot \mathrm{x}}$ 必然是方程

$$
y^{\prime \prime}+p y^{\prime}+q y=\bar{P}(x) \mathrm{e}^{(\lambda-\mathrm{i} \omega) \cdot r}
$$

的特解, 这里 $\bar{Q}_{m}$ 表示与 $Q_{m}$ 成共轭的 $m$ 次多项式.于是, 根据第六节定理 4 , 方 程 (1) 具有形如

$$
y^{\cdot}=x^{k} Q_{m} \mathrm{e}^{(\lambda+i \omega) x}+x^{k} \bar{Q}_{m} \mathrm{e}^{(\lambda-i \omega) x}
$$

的特解. 上式可写为

$$
\begin{aligned}
y^{*} & =x^{k} \mathrm{e}^{\lambda r}\left[Q_{m} \mathrm{e}^{\mathrm{i} \omega r}+\bar{Q}_{m} \mathrm{e}^{-\mathrm{i} \omega r}\right] \\
& =x^{k} \mathrm{e}^{\lambda r}\left[Q_{m}(\cos \omega x+\mathrm{i} \sin \omega x)+\bar{Q}_{m}(\cos \omega x-\mathrm{i} \sin \omega x)\right],
\end{aligned}
$$

由于括号内的两项是互成共轭的, 相加后即无虚部,所以可以写成实函数的形式

$$
y^{*}=x^{k} \mathrm{e}^{\lambda x}\left[R_{m}^{(1)}(x) \cos \omega x+R_{m}^{(2)}(x) \sin \omega x\right] \text {. }
$$

综上所述, 我们有如下结论:

如果 $f(x)=\mathrm{e}^{\lambda x}\left[P_{l}(x) \cos \omega x+P_{n}(x) \sin \omega x\right]$, 则二阶常系数非齐次线性 微分方程 (1) 的特解可设为

$$
y^{*}=x^{k} \mathrm{e}^{\lambda x}\left[R_{m !}^{(1)}(x) \cos \omega x+R_{m}^{(2)}(x) \sin \omega x\right],
$$

其中 $R_{m}^{(1)}(x) 、 R_{m}^{(2)}(x)$ 是 $m$ 次多项式, $m=\max \{l, n\}$, 而 $k$ 按 $\lambda+\mathrm{i} \omega$ （或 $\lambda-i \omega)$ 不是特征方程的根、或是特征方程的单根依次取 0 或 1 .

上述结论可推广到 $n$ 阶常系数非齐次线性微分方程, 但要注意(5)式中的 $k$ 是特征方程中含根 $\lambda+i \omega$ (或 $\lambda-i \omega$ ) 的重复次数.

例 3 求微分方程 $y^{\prime \prime}+y=x \cos 2 x$ 的一个特解.

解 所给方程是二阶常系数非齐次线性方程, 且 $f(x)$ 属于 $\mathrm{e}^{\lambda x}\left[P_{1}(x)\right.$ $\cos \omega x+P_{n}(x) \sin \omega x$ ] 型(其中 $\lambda=0, \omega=2, P_{1}(x)=x, P_{n}(x)=0$ ).

与所给方程对应的齐次方程为

$$
y^{\prime \prime}+y=0,
$$

它的特征方程为

$$
r^{2}+1=0 .
$$

由于这里 $\lambda+i \omega=2 i$ 不是特征方程的根, 所以应设特解为

$$
y^{\cdot}=(a x+b) \cos 2 x+(c x+d) \sin 2 x .
$$

把它代入所给方程,得

$$
(-3 a x-3 b+4 c) \cos 2 x-(3 c x+3 d+4 a) \sin 2 x=x \cos 2 x .
$$

比较两端同类项的系数,得

$$
\left\{\begin{array}{l}
-3 a=1, \\
-3 b+4 c=0 \\
-3 c=0, \\
-3 d-4 a=0
\end{array}\right.
$$

由此解得

$$
a=-\frac{1}{3}, b=0, c=0, d=\frac{4}{9} \text {. }
$$

于是求得一个特解为

$$
y^{\cdot}=-\frac{1}{3} x \cos 2 x+\frac{4}{9} \sin 2 x .
$$

例 4 求微分方程 $y^{\prime \prime}-y=\mathrm{e}^{x} \cos 2 x$ 的一个特解.

解 这是二阶常系数非齐次线性方程, 且 $f(x)$ 属 $\mathrm{e}^{\lambda \mathrm{r}}\left[P_{1}(x) \cos \omega x+\right.$ $P_{n}(x) \sin \omega x$ ]型(这里 $\lambda=1, \omega=2, P_{l}(x)=1, P_{n}(x)=0$ ).

特征方程为 $r^{2}-1=0$, 由于 $\lambda+i \omega=1+2 i$ 不是特征方程的根, 所以应设特 解为

$$
y^{*}=\mathrm{e}^{x}(a \cos 2 x+b \sin 2 x) .
$$

求导得

$$
\begin{aligned}
& y^{\prime \prime \prime}=\mathrm{e}^{x}[(a+2 b) \cos 2 x+(-2 a+b) \sin 2 x], \\
& y^{\prime \prime \prime}=\mathrm{e}^{x}[(-3 a+4 b) \cos 2 x+(-4 a-3 b) \sin 2 x] .
\end{aligned}
$$

代入所给方程, 得

$$
4 \mathrm{e}^{x}[(-a+b) \cos 2 x-(a+b) \sin 2 x]=\mathrm{e}^{x} \cos 2 x,
$$

比较两端同类项的系数,有

$$
\left\{\begin{array} { l } 
{ - a + b = \frac { 1 } { 4 } , } \\
{ a + b = 0 , }
\end{array} \text { 得 } \left\{\begin{array}{l}
a=-\frac{1}{8}, \\
b=\frac{1}{8} .
\end{array}\right.\right.
$$

因此所给方程的一个特解为

$$
y^{*}=\frac{1}{8} \mathrm{e}^{x}(\sin 2 x-\cos 2 x) \text {. }
$$

例 5 在第六节例 1 中, 设物体受弹性恢复力 $f$ 和铅直干扰力 $F$ 的作用. 试 求物体的运动规律.

解 这里需要求出无阻尼强迫振动方程

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+k^{2} x=h \sin p t
$$

的通解.

对应的齐次微分方程 (即无阻尼自由振动方程) 为

$$
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+k^{2} x=0,
$$

它的特征方程 $r^{2}+k^{2}=0$ 的根为 $r= \pm \mathrm{i} k$. 故方程 (7) 的通解为

$$
X=C_{1} \cos k t+C_{2} \sin k t \text {. }
$$

令

$$
C_{1}=A \sin \varphi, C_{2}=A \cos \varphi,
$$

则方程 (7)的通解又可写成

$$
X=A \sin (k t+\varphi),
$$

其中, $A, \varphi$ 为任意常数.

方程 (6)右端的函数

$$
f(t)=h \sin p t
$$

与 $f(t)=\mathrm{e}^{\lambda t}\left[P_{l}(t) \cos \omega t+P_{n}(t) \sin \omega t\right]$ 相比较, 有 $\lambda=0, \omega=p, P_{1}(t)=0$, $P_{n}(t)=h$. 现在分别就 $p \neq k$ 和 $p=k$ 两种情形讨论如下:

(i) 如果 $p \neq k$, 则 $\lambda \pm i \omega= \pm i p$ 不是特征方程的根, 故设

$$
x^{*}=a_{1} \cos p t+b_{1} \sin p t \text {. }
$$

代入方程 (6)求得

$$
a_{1}=0, \quad b_{1}=\frac{h}{k^{2}-p^{2}},
$$

于是

$$
x^{*}=\frac{h}{k^{2}-p^{2}} \sin p t \text {. }
$$

从而当 $p \neq k$ 时,方程 (6) 的通解为

$$
x=X+x^{*}=A \sin (k t+\varphi)+\frac{h}{k^{2}-p^{2}} \sin p t .
$$

上式表示, 物体的运动由两部分组成, 这两部分都是简谐振动. 上式第一项 表示息由振动,第二项所表示的振动叫做强追振动. 强迫振动是干扰力引起的, 它的角频率即是干扰力的角频率 $p$; 当干扰力的角频率 $p$ 与振动系统的固有频 率 $k$ 相差很小时,它的振幅 $\left|\frac{h}{k^{2}-p^{2}}\right|$ 可以很大.

(ii) 如果 $p=k$ 则 $\lambda \pm i \omega= \pm \mathrm{i} p$ 是特征方程的根. 故设

$$
x^{*}=t\left(a_{1} \cos k t+b_{1} \sin k t\right) \text {. }
$$

代入方程 (6)求得

于是

$$
\begin{aligned}
& a_{1}=-\frac{h}{2 k}, b_{1}=0 . \\
& x^{*}=-\frac{h}{2 k} t \cos k t .
\end{aligned}
$$

从而当 $p=k$ 时,方程 (6) 的通解为

$$
x=X+x^{*}=A \sin (k t+\varphi)-\frac{h}{2 k} t \cos k t .
$$

上式右端第二项表明, 强迫振动的振幅 $\frac{h}{2 k} t$ 随时间 $t$ 的增大而无限增大. 这 就发生所谓共振现象. 为了避免共振现象, 应使干扰力的角频率 $p$ 不要靠近振 动系统的固有频率 $k$. 反之, 如果要利用共振现象, 则应使 $p=k$ 或使 $p$ 与 $k$ 尽量 靠近.

有阻尼的强迫振动问题可作类似的讨论,这里从略了.

## 习 题 $7-8$

1. 求下列各微分方程的通解:
(1) $2 y^{\prime \prime}+y^{\prime}-y=2 \mathrm{e}^{x}$;
(2) $y^{\prime \prime}+a^{2} y=\mathrm{e}^{x}$;
(3) $2 y^{\prime \prime}+5 y^{\prime}=5 x^{2}-2 x-1$;
(4) $y^{\prime \prime}+3 y^{\prime}+2 y=3 x \mathrm{e}^{-x}$;
(5) $y^{\prime \prime}-2 y^{\prime}+5 y=\mathrm{e}^{x} \sin 2 x$;
(6) $y^{\prime \prime}-6 y^{\prime}+9 y=(x+1) \mathrm{e}^{3 . x}$;
(7) $y^{\prime \prime}+5 y^{\prime}+4 y=3-2 x$;
(8) $y^{\prime \prime}+4 y=x \cos x$;
(9) $y^{2}+y=\mathrm{e}^{r}+\cos x$;
(10) $y^{\mu}-y=\sin ^{2} x$.
2. 求下列各微分方程满足已给初始条作的特解:
(1) $y^{\prime \prime}+y+\sin 2 x=0,\left.y\right|_{, r z z}=1,\left.y^{\prime}\right|_{, r: n}=1$;
(2) $y^{\prime \prime}-3 y^{\prime}+2 y=5,\left.y\right|_{x=0}=1,\left.y^{\prime}\right|_{, x=0}=2$;
(3) $y^{\prime \prime}-10 y^{\prime}+9 y=\mathrm{e}^{2 x},\left.y\right|_{1,0}=\frac{6}{7},\left.y^{\prime}\right|_{x \rightarrow 0}=\frac{33}{7}$;
(4) $y^{\prime \prime}-y=4 . x \mathrm{e}^{x},\left.y\right|_{,=0}=0,\left.y^{\prime}\right|_{,=0}=1$;
(5) $y^{\prime \prime}-4 y^{\prime}=5,\left.y\right|_{x+11}=1,\left.y^{\prime}\right|_{x+11}=0$. 3. 大炮以仰角 $\alpha$ 、初速度 $v_{\mathfrak{1}}$ 发射炮弹, 若不计空气阻力, 求弹道曲线.
3. 在 RLC 含源串联电路中, 电动势为 $E$ 的电源对电容器 $C$ 充电。已知 $E=20 \mathrm{~V}$, $C=0.2 \mu \mathrm{F}, L=0.1 \mathrm{H}, R=1000 \Omega$, 试求台上开关 $\mathrm{S}$ 后的电流 $i(t)$ 及电压 $u_{c}(t)$.
4. 一链条息挂在一钉子上, 起动时一端离开钉子 $8 \mathrm{~m}$ 另一端离开钉子 $12 \mathrm{~m}$, 分别在以下 两种情况下求链条湖下来所需要的时间:

(1) 若不计钉子对链条所产生的摩擦力;

(2) 若摩擦力的大小等于 $1 \mathrm{~m}$ 长的链条所受需力的大小.

6. 设函数 $\varphi(x)$ 连续,且满足

$$
\varphi(x)=\mathrm{e}^{t}+\int_{11}^{x} t \varphi(t) \mathrm{d} t-x \int_{0}^{x} \varphi(t) \mathrm{d} t,
$$

求 $\varphi(x)$.

## *第九节 欧 拉 方 程

变系数的线性微分方程,一般说来都是不容易求解的. 但是有些特殊的变系 数线性微分方程,则可以通过变量代换化为常系数线性微分方程,因而容易求 解，欧拉方程就是其中的一种.

形如

$$
x^{n} y^{(n)}+p_{1} x^{n-1} y^{(n-1)}+\cdots+p_{n-1} x y^{\prime}+p_{n} y=f(x)
$$

的方程 (其中 $p_{1}, p_{2}, \cdots, p_{n}$ 为常数), 叫做欧拉方程.

作变换

$$
x=\mathrm{e}^{\prime} \text { 或 } t=\ln x,
$$

将自变量 $x$ 换成 $t^{\Phi}$, 我们有

$$
\begin{aligned}
& \frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{\mathrm{d} y}{\mathrm{~d} t} \cdot \frac{\mathrm{d} t}{\mathrm{~d} x}=\frac{1}{x} \frac{\mathrm{d} y}{\mathrm{~d} t}, \\
& \frac{\mathrm{d}^{2} y}{\mathrm{~d} x^{2}}=\frac{1}{x^{2}}\left(\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}-\frac{\mathrm{d} y}{\mathrm{~d} t}\right), \\
& \frac{\mathrm{d}^{3} y}{\mathrm{~d} x^{3}}=\frac{1}{x^{3}}\left(\frac{\mathrm{d}^{3} y}{\mathrm{~d} t^{3}}-3 \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}+2 \frac{\mathrm{d} y}{\mathrm{~d} t}\right) .
\end{aligned}
$$

如果采用记号 $\mathrm{D}$ 表示对 $\iota$ 求导的运算 $\frac{\mathrm{d}}{\mathrm{d} \ell}$, 那么上述计算结果可以写成

$$
\begin{aligned}
x y^{\prime} & =\mathrm{D} y, \\
x^{2} y^{\prime \prime} & =\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}-\frac{\mathrm{d} y}{\mathrm{~d} t}=\left(\frac{\mathrm{d}^{2}}{\mathrm{~d} t^{2}}-\frac{\mathrm{d}}{\mathrm{d} t}\right) y \\
& =\left(\mathrm{D}^{2}-\mathrm{D}\right) y=\mathrm{D}(\mathrm{D}-1) y,
\end{aligned}
$$

(1) 这里仅在 $x>0$ 范国内求解. 如果姴在 $x<0$ 内求解, 则可作变涣 $x=-\mathrm{e}^{t}$ 或 $t=\ln (-x)$. 所得结 果与 $x>0$ 内的结果相类促. 一般的,有

$$
\begin{aligned}
x^{3} y^{\prime \prime \prime} & =\frac{\mathrm{d}^{3} y}{\mathrm{~d} t^{3}}-3 \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}+2 \frac{\mathrm{d} y}{\mathrm{~d} t} \\
& =\left(\mathrm{D}^{3}-3 \mathrm{D}^{2}+2 \mathrm{D}\right) y=\mathrm{D}(\mathrm{D}-1)(\mathrm{D}-2) y,
\end{aligned}
$$

把它代入欧拉方程 (1), 便得一个以 $t$ 为自变黑的常系数线性微分方程. 在求出 这个方程的解后, 把 $t$ 换成 $\ln x$, 即得原方程的解.

例 求欧拉方程 $x^{3} y^{\prime \prime \prime}+x^{2} y^{\prime \prime}-4 x y^{\prime}=3 x^{2}$ 的通解.

解 作变换 $x=\mathrm{e}^{t}$ 或 $t=\ln x$, 原方程化为

即

$$
\mathrm{D}(\mathrm{D}-1)(\mathrm{D}-2) y+\mathrm{D}(\mathrm{D}-1) y-4 \mathrm{D} y=3 \mathrm{e}^{2 t} \text {, }
$$

或

$$
\mathrm{D}^{3} y-2 \mathrm{D}^{2} y-3 \mathrm{D} y=3 \mathrm{e}^{2 t} \text {, }
$$

$$
\frac{\mathrm{d}^{3} y}{\mathrm{~d} t^{3}}-2 \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}-3 \frac{\mathrm{d} y}{\mathrm{~d} t}=3 \mathrm{e}^{2 t^{*}} .
$$

方程 (2) 所对应的齐次方程为

$$
\frac{\mathrm{d}^{3} y}{\mathrm{~d} t^{3}}-2 \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}-3 \frac{\mathrm{d} y}{\mathrm{~d} t}=0,
$$

其特征方程为

$$
r^{3}-2 r^{2}-3 r=0,
$$

它有三个根: $r_{1}=0, r_{2}=-1, r_{3}=3$. 于是方程 (3) 的通解为

$$
Y=C_{1}+C_{2} \mathrm{e}^{-t}+C_{3} \mathrm{e}^{3 t}=C_{1}+\frac{C_{2}}{x}+C_{3} x^{3} .
$$

根据上节第一目,特解的形式为

$$
y^{*}=b \mathrm{e}^{2 t}=b x^{2},
$$

代入原方程, 求得 $b=-\frac{1}{2}$, 即

$$
y^{\cdot}=-\frac{x^{2}}{2}
$$

于是,所给欧拉方程的通解为 10

$$
y=C_{1}+\frac{C_{2}}{x}+C_{3} x^{3}-\frac{1}{2} x^{2} .
$$

## * 习 题 7-9

求下列欧拉方程的通解:

1. $x^{2} y^{\prime \prime}+x y^{\prime}-y=0$;
2. $y^{\prime \prime}-\frac{y^{\prime}}{x}+\frac{y}{x^{2}}=\frac{2}{x}$;

(1) 这是在 $x>0$ 内所求得的通解. 窎易俭证, 在 $x<0$ 内, 它也是所给方程的通解.
3. $x^{3} y^{\prime \prime \prime}+3 x^{2} y^{\prime \prime}-2 x y^{\prime}+2 y=0$;
4. $x^{2} y^{\prime \prime}-2 x y^{\prime}+2 y=\ln ^{2} x-2 \ln x$;
5. $x^{2} y^{\prime \prime}+x y^{\prime}-4 y=x^{3}$ :
6. $x^{2} y^{\prime \prime}-x y^{\prime}+4 y=x \sin (\ln x)$;
7. $x^{2} y^{\prime \prime}-3 x y^{\prime}+4 y=x+x^{2} \ln x$;
8. $x^{3} y^{\prime \prime \prime}+2 x y^{\prime}-2 y=x^{2} \ln x+3 x$.

## *第十节 常系数线性微分方程组解法举例

前面讨论的是由一个微分方程求解一个未知函数的情形. 但在研究某些实 际问题时,还会遇到由几个微分方程联立起来共同确定几个具有同一自变量的 函数的情形.这些联立的微分方程称为微分方程组.

如果微分方程组中的每一个微分方程都是常系数线性微分方程, 那么, 这种 微分方程组就叫做常系数线性微分方程组.

对于常系数线性微分方程组,我们可以用下述方法求它的解：

第一步 从方程组中消去一些未知函数及其各阶导数, 得到只含有一个未 知函数的高阶常系数线性微分方程.

第二步 解此高阶微分方程, 求出满足该方程的未知函数.

第三步 把已求得的函数代入原方程组,一般说来,不必经过积分就可求出 其余的未知函数.

例 1 解微分方程组

$$
\left\{\begin{array}{l}
\frac{\mathrm{d} y}{\mathrm{~d} x}=3 y-2 z, \\
\frac{\mathrm{d} z}{\mathrm{~d} x}=2 y-z,
\end{array}\right.
$$

解 这是含有两个未知函数 $y(x) 、 z(x)$ 的由两个一阶常系数线性方程组 成的方程组.

设法消去未知函数 $y$. 由 (2) 式得

$$
y=\frac{1}{2}\left(\frac{\mathrm{d} z}{\mathrm{~d} x}+z\right) \text {. }
$$

对上式两端求导,有

$$
\frac{\mathrm{d} y}{\mathrm{~d} x}=\frac{1}{2}\left(\frac{\mathrm{d}^{2} z}{\mathrm{~d} x^{2}}+\frac{\mathrm{d} z}{\mathrm{~d} x}\right) .
$$

把 (3)、(4)两式代入(1)式并化简,得

$$
\frac{\mathrm{d}^{2} z}{\mathrm{~d} x^{2}}-2 \frac{\mathrm{d} z}{\mathrm{~d} x}+z=0 .
$$

这是一个二阶常系数线性微分方程,它的通解是

$$
\approx=\left(C_{1}+C_{2} x\right) \mathrm{e}^{x} .
$$

再把(5).式代入(3)式,得

$$
y=\frac{1}{2}\left(2 C_{1}+C_{2}+2 C_{2} x\right) \mathrm{e}^{x} .
$$

将 (5), (6) 联立起来, 就得到所给方程组的通解.

如果我们要得到方程组满足初始条件.

$$
\left.y\right|_{x=0}=1,\left.\quad z\right|_{x=0}=0
$$

的特解, 只需将此条件代入(6)和(5)式,得

由此求得

$$
\left\{\begin{array}{l}
1=\frac{1}{2}\left(2 C_{1}+C_{2}\right), \\
0=C_{1} .
\end{array}\right.
$$

于是所给微分方程组满足上述初始条件的特解为

$$
\left\{\begin{array}{l}
y=(1+2 x) \mathrm{e}^{x}, \\
z=2 x \mathrm{e}^{x} .
\end{array}\right.
$$

在讨论常系数线性微分方程 (或方程组) 时, 常采用第七节中引人的记号 D 表示对自变量 $x$ 求导的运算 $\frac{\mathrm{d}}{\mathrm{d} x}$.

例 2 解微分方程组

$$
\left\{\begin{array}{l}
\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}+\frac{\mathrm{d} y}{\mathrm{~d} t}-x=\mathrm{e}^{\prime}, \\
\frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}+\frac{\mathrm{d} x}{\mathrm{~d} t}+y=0 .
\end{array}\right.
$$

解 用记号 $\mathrm{D}$ 表示 $\frac{\mathrm{d}}{\mathrm{d} t}$, 则方程组可记作

$$
\left\{\begin{array}{l}
\left(D^{2}-1\right) x+D y=\mathrm{e}^{t}, \\
\mathrm{D} x+\left(\mathrm{D}^{2}+1\right) y=0 .
\end{array}\right.
$$

我们可以类似于解代数方程组那样消去一个未知数, 例如为消去 $x$, 可作如 下运算：

$$
\begin{aligned}
& (7)-(8) \times \mathrm{D}:-x-\mathrm{D}^{3} y=\mathrm{e}^{\prime}, \\
& (8)+(9) \times \mathrm{D}:\left(-\mathrm{D}^{4}+\mathrm{D}^{2}+1\right) y=\mathrm{De}^{\prime},
\end{aligned}
$$

即

$$
\left(-D^{4}+D^{2}+1\right) y=e^{\prime} \text {. }
$$

(10)式为四阶非齐次线性方程, 其特征方程为

$$
-r^{4}+r^{2}+1=0 \text {, }
$$

解得特征根为

$$
r_{1,2}= \pm \alpha= \pm \sqrt{\frac{1+\sqrt{5}}{2}}, \quad r_{3.4}= \pm \mathrm{i} \beta= \pm \mathrm{i} \sqrt{\frac{\sqrt{5}-1}{2}},
$$

容易求得一个特解 $y^{*}=\mathrm{e}^{\prime}$, 于是得 (10) 的通解为

$$
y=C_{1} \mathrm{e}^{-a t}+C_{2} \mathrm{e}^{a t}+C_{3} \cos \beta t+C_{4} \sin \beta t+\mathrm{e}^{\prime} .
$$

再求 $x$. 由 (9) 式,即有

$$
x=-\mathrm{D}^{3} y-\mathrm{e}^{t},
$$

以(11)式代入上式,即得

$$
x=\alpha^{3} C_{1} \mathrm{e}^{a t}-\alpha^{3} C_{2} \mathrm{e}^{a t}-\beta^{3} C_{3} \sin \beta t+\beta^{3} C_{4} \cos \beta t-2 \mathrm{e}^{t} .
$$

将 (11) 和 (12) 两个函数联立, 就是所求方程组的通解.

这里要注意,在求得一个未知函数以后,再求另一个未知函数时,一般不再 积分 (积分就会出现新的任意常数, 从(11)、(12) 两式可知两式中的任意常数之 间有着确定的关系).

我们也可用行列式解上述方程组.由(7)和 (8), 有

$$
\left|\begin{array}{cc}
D^{2}-1 & D \\
D & D^{2}+1
\end{array}\right| y=\left|\begin{array}{cc}
D^{2}-1 & e^{\prime} \\
D & 0
\end{array}\right| \text {, }
$$

即

$$
\left(D^{4}-D^{2}-1\right) y=-\mathrm{e}^{t} \text {. }
$$

这与 $(10)$ 式是一样的. 但再求 $x$ 时, 不宜再次应用行列式. 如再应用行列式, 得

即

$$
\left|\begin{array}{cc}
\mathrm{D}^{2}-1 & \mathrm{D} \\
\mathrm{D} & \mathrm{D}^{2}+1
\end{array}\right| x=\left|\begin{array}{cc}
\mathrm{e}^{\prime} & \mathrm{D} \\
0 & \mathrm{D}^{2}+1
\end{array}\right| \text {, }
$$

解得

$$
\left(D^{4}-D^{2}-1\right) x=2 e^{\prime} \text {, }
$$

$$
x=A_{1} \mathrm{e}^{-{ }^{a t}}+A_{2} \mathrm{e}^{a t}+A_{3} \cos \beta t+A_{4} \sin \beta t-2 \mathrm{e}^{t},
$$

则必须说明 $A_{1} 、 A_{2} 、 A_{3} 、 A_{4}$ 与 $C_{1} 、 C_{2} 、 C_{3} 、 C_{4}$ 之间的关系.

注意这里的“系数行列式”

$$
\left|\begin{array}{cc}
D^{2}-1 & D \\
D & D^{2}+1
\end{array}\right|=D^{4}-D^{2}-1
$$

是 $\mathrm{D}$ 的四次多项式, 这就标志着微分方程组是四阶的, 它的通解中一定恰含四 个任意常数.

## * 习 题 7-10

1. 求下列微分方程组的通解:
(1) $\left\{\begin{array}{l}\frac{\mathrm{d} y}{\mathrm{~d} x}=z, \\ \frac{\mathrm{d} z}{\mathrm{~d} x}=y \text {; }\end{array}\right.$
(2) $\left\{\begin{array}{l}\frac{\mathrm{d}^{2} x}{\mathrm{~d} t^{2}}=y \\ \frac{\mathrm{d}^{2} y}{\mathrm{~d} t^{2}}=x ;\end{array}\right.$
(3) $\left\{\begin{array}{l}\frac{\mathrm{d} x}{\mathrm{~d} t}+\frac{\mathrm{d} y}{\mathrm{~d} t}=-x+y+3 \\ \frac{\mathrm{d} x}{\mathrm{~d} t}-\frac{\mathrm{d} y}{\mathrm{~d} t}=x+y-3 ;\end{array}\right.$
(4) $\left\{\begin{array}{l}\frac{\mathrm{d} x}{\mathrm{~d} t}+5 x+y=\mathrm{e}^{t} \\ \frac{\mathrm{d} y}{\mathrm{~d} t}-x-3 y=\mathrm{e}^{2 t}\end{array}\right.$

## 附录 I 二阶和三阶行列式简介

给出二元线性方程组

求这方程组的解.

$$
\left\{\begin{array}{l}
a_{11} x_{1}+a_{12} x_{2}=b_{1}, \\
a_{21} x_{1}+a_{22} x_{2}=b_{2},
\end{array}\right.
$$

用大家熟知的消元法, 分别消去方程组 (1) 中的 $x_{2}$ 及 $x_{1}$, 得

$$
\left\{\begin{array}{l}
\left(a_{11} a_{22}-a_{12} a_{21}\right) x_{1}=b_{1} a_{22}-a_{12} b_{2}, \\
\left(a_{11} a_{22}-a_{12} a_{21}\right) x_{2}=a_{11} b_{2}-b_{1} a_{21} .
\end{array}\right.
$$

下面引人二阶行列式, 然后利用二阶行列式来进一步讨论上述问题.

设已知四个数排成正方形表

$$
\left[\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right),
$$

则数 $a_{11} a_{22}-a_{12} a_{21}$ 称为对应于这个表的二阶行列式, 用记号

$$
\left|\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right|
$$

表示, 因此

$$
\left|\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right|=a_{11} a_{22}-a_{12} a_{21} .
$$

数 $a_{11}, a_{12}, a_{21}, a_{22}$ 叫做行列式 (3) 的元甞, 横排叫做行, 坚排叫做烈. 元素 $a_{i j}$ 中的第一个指标 $i$ 和第二个指标 $j$, 依次表示行数和列数. 例如, 元素 $a_{21}$ 在行 列式(3)中位于第二行和第一列.

现在,方程组(2)可利用行列式来表示. 设

$$
\begin{aligned}
D & =\left|\begin{array}{ll}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{array}\right|=a_{11} a_{22}-a_{12} a_{21}, \\
D_{1} & =\left|\begin{array}{ll}
b_{1} & a_{12} \\
b_{2} & a_{22}
\end{array}\right|=b_{1} a_{22}-a_{12} b_{2}, \\
D_{2} & =\left|\begin{array}{ll}
a_{11} & b_{1} \\
a_{21} & b_{2}
\end{array}\right|=a_{11} b_{2}-b_{1} a_{21},
\end{aligned}
$$

则方程组 (2) 可写成

$$
\left\{\begin{array}{l}
D x_{1}=D_{1} \\
D x_{2}=D_{2}
\end{array}\right.
$$

我们注意到, $D$ 就是方程组 (1)中 $x_{1}$ 及 $x_{2}$ 的系数构成的行列式,因此称为 系数行列式,而 $D_{1}$ 和 $D_{2}$ 分别是用方程组 (1)右端的常数项代替 $D$ 的第一列和 第二列而形成的.

若 $D \neq 0$, 则方程组 (2) 的解为

$$
x_{1}=\frac{D_{1}}{D}, x_{2}=\frac{D_{2}}{D} \text {. }
$$

把 (4) 中 $x_{1}$ 及 $x_{2}$ 的值代入方程组 (1), 便可证实 $x_{1}$ 及 $x_{2}$ 的这对值也是方 程组 (1) 的解. 另一方面, (2) 是由 (1) 导出的, 因此 (1) 的解一定是 (2) 的解. 现在 (2) 只有一组解 (4), 所以 (4) 是方程组 (1) 的唯一解. 由此得出结论:

在 $D \neq 0$ 的条件下,方程组 (1) 有唯一的解

$$
x_{1}=\frac{D_{1}}{D}, x_{2}=\frac{D_{2}}{D} \text {. }
$$

例 1 解方程组

$$
\left\{\begin{array}{l}
2 x+3 y=8 \\
x-2 y=-3
\end{array}\right.
$$

解 $D=\left|\begin{array}{cc}2 & 3 \\ 1 & -2\end{array}\right|=2 \times(-2)-3 \times 1=-7$,

$$
\begin{aligned}
& D_{1}=\left|\begin{array}{cc}
8 & 3 \\
-3 & -2
\end{array}\right|=8 \times(-2)-3 \times(-3)=-7, \\
& D_{2}=\left|\begin{array}{cc}
2 & 8 \\
1 & -3
\end{array}\right|=2 \times(-3)-8 \times 1=-14 .
\end{aligned}
$$

因 $D=-7 \neq 0$, 故所给方程组有唯一解

$$
x=\frac{D_{1}}{D}=\frac{-7}{-7}=1, \quad y=\frac{D_{2}}{D}=\frac{-14}{-7}=2 .
$$

下面介绍三阶行列式概念.

设已知九个数排成正方形表

$$
\left(\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right),
$$

则数 $a_{11} a_{22} a_{33}+a_{12} a_{23} a_{31}+a_{13} a_{21} a_{32}-a_{13} a_{22} a_{31}-a_{12} a_{21} a_{33}-a_{11} a_{23} a_{32}$ 称为 对应于这个表的三阶行列式,用记号

表示, 因此

$$
\left|\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{32} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right|
$$

关于三阶行列式的元素、行、列等概念, 与二阶行列式的相应概念类似,不再 重复.

(5)式右端相当复杂, 我们可以啃助下列图形得出它的计算法则 (通常称为 对角线法则):

$+$

- 

行列式中从左上角到右下角的直线称为主对角线, 从右上角到左下角的直 线称为次对角线. 主对角线上元素的乘积以及位于主对角线的平行线上的元素 与对角上的元素的乘积,前面都取正号. 次对角线上元素的乘积以及位于次对角 线的平行线上的元索与对角上的元素的乘积,前面都取负号.

$$
\begin{aligned}
& \text { 例 } 2\left|\begin{array}{ccc}
2 & 1 & 2 \\
-4 & 3 & 1 \\
2 & 3 & 5
\end{array}\right| \\
& =2 \times 3 \times 5+1 \times 1 \times 2+2 \times(-4) \times 3- \\
& 2 \times 3 \times 2-1 \times(-4) \times 5-2 \times 1 \times 3 \\
& =30+2-24-12+20-6=10 \text {. }
\end{aligned}
$$

利用交换律及结合律, 可把(5)式改写如下:

$$
\left|\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right|=\begin{gathered}
a_{11}\left(a_{22} a_{33}-a_{23} a_{32}\right)-a_{12}\left(a_{21} a_{33}-a_{23} a_{31}\right)+ \\
a_{13}\left(a_{21} a_{32}-a_{22} a_{31}\right) .
\end{gathered}
$$

把上式右端三个括号中的式子表示为二阶行列式,则有

$$
\left|\begin{array}{lll}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33}
\end{array}\right|=a_{11}\left|\begin{array}{ll}
a_{22} & a_{23} \\
a_{32} & a_{33}
\end{array}\right|-a_{12}\left|\begin{array}{ll}
a_{21} & a_{23} \\
a_{31} & a_{33}
\end{array}\right|+a_{13}\left|\begin{array}{ll}
a_{21} & a_{22} \\
a_{31} & a_{32}
\end{array}\right| .
$$

上式称为三阶行列式按第一行的展开式.

例 3 将例 2 中的行列式按第一行展开并计算它的值.

$$
\text { 解 } \begin{aligned}
\left|\begin{array}{ccc}
2 & 1 & 2 \\
-4 & 3 & 1 \\
2 & 3 & 5
\end{array}\right| & =2\left|\begin{array}{ll}
3 & 1 \\
3 & 5
\end{array}\right|-\left|\begin{array}{cc}
-4 & 1 \\
2 & 5
\end{array}\right|+2\left|\begin{array}{cc}
-4 & 3 \\
2 & 3
\end{array}\right| \\
& =2 \times 12-(-22)+2 \times(-18) \\
& =24+22-36=10 .
\end{aligned}
$$

## 习 题

1. 利用二价行列式解下列方程组:
(1) $\left\{\begin{array}{l}5 x-y=2, \\ 3 x+2 y=9 ;\end{array}\right.$
(2) $\left\{\begin{array}{l}3 x+4 y=2, \\ 2 x+3 y=7\end{array}\right.$
2. 利用对角线法则，计算下列各行列式:
(1) $\left|\begin{array}{ccc}2 & 0 & 1 \\ 1 & -4 & -1 \\ -1 & 8 & 3\end{array}\right|$;
(2) $\left|\begin{array}{ccc}4 & -2 & 4 \\ 10 & 2 & 12 \\ 1 & 2 & 2\end{array}\right|$;
(3) $\left|\begin{array}{lll}3 & 4 & 2 \\ 7 & 5 & 1 \\ 3 & 2 & 4\end{array}\right|$;
(4) $\left|\begin{array}{ccc}1 & 1 & 1 \\ 1 & 1+a & 1 \\ 1 & 1 & 1+b\end{array}\right|$.
3. 将下列行列式接第一行展开并计其它们的值:
(1) $\left|\begin{array}{lll}1 & 2 & 3 \\ 3 & 1 & 2 \\ 2 & 3 & 1\end{array}\right|$;
(2) $\left|\begin{array}{ccc}-1 & 2 & 2 \\ 2 & -1 & 2 \\ 2 & 2 & -1\end{array}\right|$.
4. 证明下列等式:
(1) $\left|\begin{array}{lll}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{array}\right|=-a_{21}\left|\begin{array}{ll}a_{12} & a_{13} \\ a_{32} & a_{33}\end{array}\right|+a_{22}\left|\begin{array}{ll}a_{11} & a_{13} \\ a_{31} & a_{33}\end{array}\right|-a_{23}\left|\begin{array}{ll}a_{11} & a_{12} \\ a_{31} & a_{32}\end{array}\right|$;
(2) $\left|\begin{array}{lll}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{array}\right|=a_{31}\left|\begin{array}{ll}a_{12} & a_{13} \\ a_{22} & a_{23}\end{array}\right|-a_{32}\left|\begin{array}{ll}a_{11} & a_{13} \\ a_{21} & a_{23}\end{array}\right|+a_{33}\left|\begin{array}{ll}a_{11} & a_{12} \\ a_{21} & a_{22}\end{array}\right| ;$

[注: 上面这两个等式分别称为三阶行列式按第二行利技第三行的展开式.]

## 附录 III 积 分 表

## $($ 一) 含有 $a x+b$ 的积分

1. $\int \frac{\mathrm{d} x}{a x+b}=\frac{1}{a} \ln |a x+b|+C$
2. $\int(a x+b)^{\mu} \mathrm{d} x=\frac{1}{a(\mu+1)}(a x+b)^{\mu+1}+C(\mu \neq-1)$
3. $\int \frac{x}{a x+b} \mathrm{~d} x=\frac{1}{a^{2}}(a x+b-b \ln |a x+b|)+C$
4. $\int \frac{x^{2}}{a x+b} \mathrm{~d} x=\frac{1}{a^{3}}\left[\frac{1}{2}(a x+b)^{2}-2 b(a x+b)+b^{2} \ln |a x+b|\right]+C$
5. $\int \frac{\mathrm{d} x}{x(a x+b)}=-\frac{1}{b} \ln \left|\frac{a x+b}{x}\right|+C$
6. $\int \frac{\mathrm{d} x}{x^{2}(a x+b)}=-\frac{1}{b x}+\frac{a}{b^{2}} \ln \left|\frac{a x+b}{x}\right|+C$
7. $\int \frac{x}{(a x+b)^{2}} \mathrm{~d} x=\frac{1}{a^{2}}\left(\ln |a x+b|+\frac{b}{a x+b}\right)+C$
8. $\int \frac{x^{2}}{(a x+b)^{2}} \mathrm{~d} x=\frac{1}{a^{3}}\left(a x+b-2 b \ln |a x+b|-\frac{b^{2}}{a x+b}\right)+C$
9. $\int \frac{\mathrm{d} x}{x(a x+b)^{2}}=\frac{1}{b(a x+b)}-\frac{1}{b^{2}} \ln \left|\frac{a x+b}{x}\right|+C$

## (二) 含有 $\sqrt{a x+b}$ 的积分

10. $\int \sqrt{a x+b} \mathrm{~d} x=\frac{2}{3 a} \sqrt{(a x+b)^{3}}+C$
11. $\int x \sqrt{a x+b} \mathrm{~d} x=\frac{2}{15 a^{2}}(3 a x-2 b) \sqrt{(a x+b)^{3}}+C$
12. $\int x^{2} \sqrt{a x+b} \mathrm{~d} x=\frac{2}{105 a^{3}}\left(15 a^{2} x^{2}-12 a b x+8 b^{2}\right) \sqrt{(a x+b)^{3}}+C$
13. $\int \frac{x}{\sqrt{a x+b}} \mathrm{~d} x=\frac{2}{3 a^{2}}(a x-2 b) \sqrt{a x+b}+C$
14. $\int \frac{x^{2}}{\sqrt{a x+b}} \mathrm{~d} x=\frac{2}{15 a^{3}}\left(3 a^{2} x^{2}-4 a b x+8 b^{2}\right) \sqrt{a x+b}+C$ 15. $\int \frac{\mathrm{d} x}{x \sqrt{a x+b}}=\left\{\begin{array}{l}\frac{1}{\sqrt{b}} \ln \left|\frac{\sqrt{a x+b}-\sqrt{b}}{\sqrt{a x+b}+\sqrt{b}}\right|+C \quad(b>0) \\ \frac{2}{\sqrt{-b}} \arctan \sqrt{\frac{a x+b}{-b}+C \quad(b<0)}\end{array}\right.$
15. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{a x+b}}=-\frac{\sqrt{a x+b}}{b x}-\frac{a}{2 b} \int \frac{\mathrm{d} x}{x \sqrt{a x+b}}$
16. $\int \frac{\sqrt{a x+b}}{x} \mathrm{~d} x=2 \sqrt{a x+b}+b \int \frac{\mathrm{d} x}{x \sqrt{a x+b}}$
17. $\int \frac{\sqrt{a x+b}}{x^{2}} \mathrm{~d} x=-\frac{\sqrt{a x+b}}{x}+\frac{a}{2} \int \frac{\mathrm{d} x}{x \sqrt{a x+b}}$

## (三) 含有 $x^{2} \pm a^{2}$ 的积分

19. $\int \frac{\mathrm{d} x}{x^{2}+a^{2}}=\frac{1}{a} \arctan \frac{x}{a}+C$
20. $\int \frac{\mathrm{d} x}{\left(x^{2}+a^{2}\right)^{n}}=\frac{x}{2(n-1) a^{2}\left(x^{2}+a^{2}\right)^{n-1}}+\frac{2 n-3}{2(n-1) a^{2}} \int \frac{\mathrm{d} x}{\left(x^{2}+a^{2}\right)^{n-1}}$
21. $\int \frac{\mathrm{d} x}{x^{2}-a^{2}}=\frac{1}{2 a} \ln \left|\frac{x-a}{x+a}\right|+C$

(四) 含有 $a x^{2}+b(a>0)$ 的积分

22. $\int \frac{\mathrm{d} x}{a x^{2}+b}=\left\{\begin{array}{l}\frac{1}{\sqrt{a b}} \arctan \sqrt{\frac{a}{b}} x+C \quad(b>0) \\ \frac{1}{2 \sqrt{-a b}} \ln \left|\frac{\sqrt{a} x-\sqrt{-b}}{\sqrt{a} x+\sqrt{-b}}\right|+C \quad(b<0)\end{array}\right.$
23. $\int \frac{x}{a x^{2}+b} \mathrm{~d} x=\frac{1}{2 a} \ln \left|a x^{2}+b\right|+C$
24. $\int \frac{x^{2}}{a x^{2}+b} \mathrm{~d} x=\frac{x}{a}-\frac{b}{a} \int \frac{\mathrm{d} x}{a x^{2}+b}$
25. $\int \frac{\mathrm{d} x}{x\left(a x^{2}+b\right)}=\frac{1}{2 b} \ln \frac{x^{2}}{\left|a x^{2}+b\right|}+C$
26. $\int \frac{\mathrm{d} x}{x^{2}\left(a x^{2}+b\right)}=-\frac{1}{b x}-\frac{a}{b} \int \frac{\mathrm{d} x}{a x^{2}+b}$
27. $\int \frac{\mathrm{d} x}{x^{3}\left(a x^{2}+b\right)}=\frac{a}{2 b^{2}} \ln \frac{\left|a x^{2}+b\right|}{x^{2}}-\frac{1}{2 b x^{2}}+C$
28. $\int \frac{\mathrm{d} x}{\left(a x^{2}+b\right)^{2}}=\frac{x}{2 b\left(a x^{2}+b\right)}+\frac{1}{2 b} \int \frac{\mathrm{d} x}{a x^{2}+b}$ （五）含有 $a x^{2}+b x+c(a>0)$ 的积分
29. $\int \frac{\mathrm{d} x}{a x^{2}+b x+c}=\left\{\begin{array}{l}\frac{2}{\sqrt{4 a c-b^{2}}} \arctan \frac{2 a x+b}{\sqrt{4 a c-b^{2}}}+C \quad\left(b^{2}<4 a c\right) \\ \frac{1}{\sqrt{b^{2}-4 a c}} \ln \left|\frac{2 a x+b-\sqrt{b^{2}-4 a c}}{2 a x+b+\sqrt{b^{2}-4 a c}}\right|+C \quad\left(b^{2}>4 a c\right)\end{array}\right.$
30. $\int \frac{x}{a x^{2}+b x+c} \mathrm{~d} x=\frac{1}{2 a} \ln \left|a x^{2}+b x+c\right|-\frac{b}{2 a} \int \frac{\mathrm{d} x}{a x^{2}+b x+c}$

## (六) 含有 $\sqrt{x^{2}+a^{2}}(a>0)$ 的积分

31. $\int \frac{\mathrm{d} x}{\sqrt{x^{2}+a^{2}}}=\operatorname{arsh} \frac{x}{a}+C_{1}=\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
32. $\int \frac{\mathrm{d} x}{\sqrt{\left(x^{2}+a^{2}\right)^{3}}}=\frac{x}{a^{2} \sqrt{x^{2}+a^{2}}}+C$
33. $\int \frac{x}{\sqrt{x^{2}+a^{2}}} \mathrm{~d} x=\sqrt{x^{2}+a^{2}}+C$
34. $\int \frac{x}{\sqrt{\left(x^{2}+a^{2}\right)^{3}}} \mathrm{~d} x=-\frac{1}{\sqrt{x^{2}+a^{2}}}+C$
35. $\int \frac{x^{2}}{\sqrt{x^{2}+a^{2}}} \mathrm{~d} x=\frac{x}{2} \sqrt{x^{2}+a^{2}}-\frac{a^{2}}{2} \ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
36. $\int \frac{x^{2}}{\sqrt{\left(x^{2}+a^{2}\right)^{3}}} \mathrm{~d} x=-\frac{x}{\sqrt{x^{2}+a^{2}}}+\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
37. $\int \frac{\mathrm{d} x}{x \sqrt{x^{2}+a^{2}}}=\frac{1}{a} \ln \frac{\sqrt{x^{2}+a^{2}}-a}{|x|}+C$
38. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{x^{2}+a^{2}}}=-\frac{\sqrt{x^{2}+a^{2}}}{a^{2} x}+C$
39. $\int \sqrt{x^{2}+a^{2}} \mathrm{~d} x=\frac{x}{2} \sqrt{x^{2}+a^{2}}+\frac{a^{2}}{2} \ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
40. $\int \sqrt{\left(x^{2}+a^{2}\right)^{3}} \mathrm{~d} x=\frac{x}{8}\left(2 x^{2}+5 a^{2}\right) \sqrt{x^{2}+a^{2}}+\frac{3}{8} a^{4} \ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
41. $\int x \sqrt{x^{2}+a^{2}} \mathrm{~d} x=\frac{1}{3} \sqrt{\left(x^{2}+a^{2}\right)^{3}}+C$
42. $\int x^{2} \sqrt{x^{2}+a^{2}} \mathrm{~d} x=\frac{x}{8}\left(2 x^{2}+a^{2}\right) \sqrt{x^{2}+a^{2}}-\frac{a^{4}}{8} \ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$
43. $\int \frac{\sqrt{x^{2}+a^{2}}}{x} \mathrm{~d} x=\sqrt{x^{2}+a^{2}}+a \ln \frac{\sqrt{x^{2}+a^{2}}-a}{|x|}+C$ 44. $\int \frac{\sqrt{x^{2}+a^{2}}}{x^{2}} \mathrm{~d} x=-\frac{\sqrt{x^{2}+a^{2}}}{x}+\ln \left(x+\sqrt{x^{2}+a^{2}}\right)+C$

（七）含有 $\sqrt{x^{2}-a^{2}}(a>0)$ 的积分

45. $\int \frac{\mathrm{d} x}{\sqrt{x^{2}-a^{2}}}=\frac{x}{|x|} \operatorname{arch} \frac{|x|}{a}+C_{1}=\ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
46. $\int \frac{\mathrm{d} x}{\sqrt{\left(x^{2}-a^{2}\right)^{3}}}=-\frac{x}{a^{2} \sqrt{x^{2}-a^{2}}}+C$
47. $\int \frac{x}{\sqrt{x^{2}-a^{2}}} \mathrm{~d} x=\sqrt{x^{2}-a^{2}}+C$
48. $\int \frac{x}{\sqrt{\left(x^{2}-a^{2}\right)^{3}}} \mathrm{~d} x=-\frac{1}{\sqrt{x^{2}-a^{2}}}+C$
49. $\int \frac{x^{2}}{\sqrt{x^{2}-a^{2}}} \mathrm{~d} x=\frac{x}{2} \sqrt{x^{2}-a^{2}}+\frac{a^{2}}{2} \ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
50. $\int \frac{x^{2}}{\sqrt{\left(x^{2}-a^{2}\right)^{3}}} \mathrm{~d} x=-\frac{x}{\sqrt{x^{2}-a^{2}}}+\ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
51. $\int \frac{\mathrm{d} x}{x \sqrt{x^{2}-a^{2}}}=\frac{1}{a} \arccos \frac{a}{|x|}+C$
52. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{x^{2}-a^{2}}}=\frac{\sqrt{x^{2}-a^{2}}}{a^{2} x}+C$
53. $\int \sqrt{x^{2}-a^{2}} \mathrm{~d} x=\frac{x}{2} \sqrt{x^{2}-a^{2}}-\frac{a^{2}}{2} \ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
54. $\int \sqrt{\left(x^{2}-a^{2}\right)^{3}} \mathrm{~d} x=\frac{x}{8}\left(2 x^{2}-5 a^{2}\right) \sqrt{x^{2}-a^{2}}+\frac{3}{8} a^{4} \ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
55. $\int x \sqrt{x^{2}-a^{2}} \mathrm{~d} x=\frac{1}{3} \sqrt{\left(x^{2}-a^{2}\right)^{3}}+C$
56. $\int x^{2} \sqrt{x^{2}-a^{2}} \mathrm{~d} x=\frac{x}{8}\left(2 x^{2}-a^{2}\right) \sqrt{x^{2}-a^{2}}-\frac{a^{4}}{8} \ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$
57. $\int \frac{\sqrt{x^{2}-a^{2}}}{x} \mathrm{~d} x=\sqrt{x^{2}-a^{2}}-a \arccos \frac{a}{|x|}+C$
58. $\int \frac{\sqrt{x^{2}-a^{2}}}{x^{2}} \mathrm{~d} x=-\frac{\sqrt{x^{2}-a^{2}}}{x}+\ln \left|x+\sqrt{x^{2}-a^{2}}\right|+C$

(八) 含有 $\sqrt{a^{2}-x^{2}}(a>0)$ 的积分

59. $\int \frac{\mathrm{d} x}{\sqrt{a^{2}-x^{2}}}=\arcsin \frac{x}{a}+C$ 60. $\int \frac{\mathrm{d} x}{\sqrt{\left(a^{2}-x^{2}\right)^{3}}}=\frac{x}{a^{2} \sqrt{a^{2}-x^{2}}}+C$
60. $\int \frac{x}{\sqrt{a^{2}-x^{2}}} \mathrm{~d} x=-\sqrt{a^{2}-x^{2}}+C$
61. $\int \frac{x}{\sqrt{\left(a^{2}-x^{2}\right)^{3}}} \mathrm{~d} x=\frac{1}{\sqrt{a^{2}-x^{2}}}+C$
62. $\int \frac{x^{2}}{\sqrt{a^{2}-x^{2}}} \mathrm{~d} x=-\frac{x}{2} \sqrt{a^{2}-x^{2}}+\frac{a^{2}}{2} \arcsin \frac{x}{a}+C$
63. $\int \frac{x^{2}}{\sqrt{\left(a^{2}-x^{2}\right)^{3}}} \mathrm{~d} x=\frac{x}{\sqrt{a^{2}-x^{2}}}-\arcsin \frac{x}{a}+C$
64. $\int \frac{\mathrm{d} x}{x \sqrt{a^{2}-x^{2}}}=\frac{1}{a} \ln \frac{a-\sqrt{a^{2}-x^{2}}}{|x|}+C$
65. $\int \frac{\mathrm{d} x}{x^{2} \sqrt{a^{2}-x^{2}}}=-\frac{\sqrt{a^{2}-x^{2}}}{a^{2} x}+C$
66. $\int \sqrt{a^{2}-x^{2}} \mathrm{~d} x=\frac{x}{2} \sqrt{a^{2}-x^{2}}+\frac{a^{2}}{2} \arcsin \frac{x}{a}+C$
67. $\int \sqrt{\left(a^{2}-x^{2}\right)^{3}} \mathrm{~d} x=\frac{x}{8}\left(5 a^{2}-2 x^{2}\right) \sqrt{a^{2}-x^{2}}+\frac{3}{8} a^{4} \arcsin \frac{x}{a}+C$
68. $\int x \sqrt{a^{2}-x^{2}} \mathrm{~d} x=-\frac{1}{3} \sqrt{\left(a^{2}-x^{2}\right)^{3}}+C$
69. $\int x^{2} \sqrt{a^{2}-x^{2}} \mathrm{~d} x=\frac{x}{8}\left(2 x^{2}-a^{2}\right) \sqrt{a^{2}-x^{2}}+\frac{a^{4}}{8} \arcsin \frac{x}{a}+C$
70. $\int \frac{\sqrt{a^{2}-x^{2}}}{x} \mathrm{~d} x=\sqrt{a^{2}-x^{2}}+a \ln \frac{a-\sqrt{a^{2}-x^{2}}}{|x|}+C$
71. $\int \frac{\sqrt{a^{2}-x^{2}}}{x^{2}} \mathrm{~d} x=-\frac{\sqrt{a^{2}-x^{2}}}{x}-\arcsin \frac{x}{a}+C$

（九）含有 $\sqrt{ \pm a x^{2}+b x+c}(a>0)$ 的积分

73. $\int \frac{\mathrm{d} x}{\sqrt{a x^{2}+b x+c}}=\frac{1}{\sqrt{a}} \ln \left|2 a x+b+2 \sqrt{a} \sqrt{a x^{2}+b x+c}\right|+C$
74. $\int \sqrt{a x^{2}+b x+c} \mathrm{~d} x=\frac{2 a x+b}{4 a} \sqrt{a x^{2}+b x+c}+$

$$
\frac{4 a c-b^{2}}{8 \sqrt{a^{3}}} \ln \left|2 a x+b+2 \sqrt{a} \sqrt{a x^{2}+b x+c}\right|+C
$$

75. $\int \frac{x}{\sqrt{a x^{2}+b x+c}} \mathrm{~d} x=\frac{1}{a} \sqrt{a x^{2}+b x+c}-$

$$
\frac{b}{2 \sqrt{a^{3}}} \ln \left|2 a x+b+2 \sqrt{a} \sqrt{a x^{2}+b x+c}\right|+C
$$

76. $\int \frac{\mathrm{d} x}{\sqrt{c+b x-a x^{2}}}=-\frac{1}{\sqrt{a}} \arcsin \frac{2 a x-b}{\sqrt{b^{2}+4 a c}}+C$
77. $\int \sqrt{c+b x-a x^{2}} \mathrm{~d} x=\frac{2 a x-b}{4 a} \sqrt{c+b x-a x^{2}}+$

$$
\frac{b^{2}+4 a c}{8 \sqrt{a^{3}}} \arcsin \frac{2 a x-b}{\sqrt{b^{2}+4 a c}}+C
$$

78. $\int \frac{x}{\sqrt{c+b x-a x^{2}}} \mathrm{~d} x=-\frac{1}{a} \sqrt{c+b x-a x^{2}}+\frac{b}{2 \sqrt{a^{3}}} \arcsin \frac{2 a x-b}{\sqrt{b^{2}+4 a c}}+C$

(十) 含有 $\sqrt{ \pm \frac{x-a}{x-b}}$ 或 $\sqrt{(x-a)(b-x)}$ 的积分

79. $\int \sqrt{\frac{x-a}{x-b}} \mathrm{~d} x=(x-b) \sqrt{\frac{x-a}{x-b}}+(b-a) \ln (\sqrt{|x-a|}+\sqrt{|x-b|})+C$
80. $\int \sqrt{\frac{x-a}{b-x}} \mathrm{~d} x=(x-b) \sqrt{\frac{x-a}{b-x}}+(b-a) \arcsin \sqrt{\frac{x-a}{b-a}}+C$
81. $\int \frac{\mathrm{d} x}{\sqrt{(x-a)(b-x)}}=2 \arcsin \sqrt{\frac{x-a}{b-a}}+C(a<b)$
82. $\int \sqrt{(x-a)(b-x)} \mathrm{d} x=\frac{2 x-a-b}{4} \sqrt{(x-a)(b-x)}+$

$$
\frac{(b-a)^{2}}{4} \arcsin \sqrt{\frac{x-a}{b-a}}+C(a<b)
$$

## (十一) 含有三角函数的积分

83. $\int \sin x \mathrm{~d} x=-\cos x+C$
84. $\int \cos x \mathrm{~d} x=\sin x+C$
85. $\int \tan x \mathrm{~d} x=-\ln |\cos x|+C$
86. $\int \cot x \mathrm{~d} x=\ln |\sin x|+C$
87. $\int \sec x \mathrm{~d} x=\ln \left|\tan \left(\frac{\pi}{4}+\frac{x}{2}\right)\right|+C=\ln |\sec x+\tan x|+C$
88. $\int \csc x \mathrm{~d} x=\ln \left|\tan \frac{x}{2}\right|+C=\ln |\csc x-\cot x|+C$
89. $\int \sec ^{2} x \mathrm{~d} x=\tan x+C$ 90. $\int \csc ^{2} x \mathrm{~d} x=-\cot x+C$
90. $\int \sec x \tan x \mathrm{~d} x=\sec x+C$
91. $\int \csc x \cot x \mathrm{~d} x=-\csc x+C$
92. $\int \sin ^{2} x \mathrm{~d} x=\frac{x}{2}-\frac{1}{4} \sin 2 x+C$
93. $\int \cos ^{2} x \mathrm{~d} x=\frac{x}{2}+\frac{1}{4} \sin 2 x+C$
94. $\int \sin ^{n} x \mathrm{~d} x=-\frac{1}{n} \sin ^{n-1} x \cos x+\frac{n-1}{n} \int \sin ^{n-2} x \mathrm{~d} x$
95. $\int \cos ^{n} x \mathrm{~d} x=\frac{1}{n} \cos ^{n-1} x \sin x+\frac{n-1}{n} \int \cos ^{n-2} x \mathrm{~d} x$
96. $\int \frac{\mathrm{d} x}{\sin ^{n} x}=-\frac{1}{n-1} \cdot \frac{\cos x}{\sin ^{n-1} x}+\frac{n-2}{n-1} \int \frac{\mathrm{d} x}{\sin ^{n-2} x}$
97. $\int \frac{\mathrm{d} x}{\cos ^{n} x}=\frac{1}{n-1} \cdot \frac{\sin x}{\cos ^{n-1} x}+\frac{n-2}{n-1} \int \frac{\mathrm{d} x}{\cos ^{n-2} x}$
98. $\int \cos ^{\prime n} x \sin ^{n} x \mathrm{~d} x=\frac{1}{m+n} \cos ^{m-1} x \sin ^{n+1} x+\frac{m-1}{m+n} \int \cos ^{m-2} x \sin ^{n} x \mathrm{~d} x$

$$
=-\frac{1}{m+n} \cos ^{m+1} x \sin ^{n-1} x+\frac{n-1}{m+n} \int \cos ^{m \prime} x \sin ^{n-2} x \mathrm{~d} x
$$

100. $\int \sin a x \cos b x \mathrm{~d} x=-\frac{1}{2(a+b)} \cos (a+b) x-\frac{1}{2(a-b)} \cos (a-b) x+C$
101. $\int \sin a x \sin b x \mathrm{~d} x=-\frac{1}{2(a+b)} \sin (a+b) x+\frac{1}{2(a-b)} \sin (a-b) x+C$
102. $\int \cos a x \cos b x \mathrm{~d} x=\frac{1}{2(a+b)} \sin (a+b) x+\frac{1}{2(a-b)} \sin (a-b) x+C$
103. $\int \frac{\mathrm{d} x}{a+b \sin x}=\frac{2}{\sqrt{a^{2}-b^{2}}} \arctan \frac{a \tan \frac{x}{2}+b}{\sqrt{a^{2}-b^{2}}}+C\left(a^{2}>b^{2}\right)$
104. $\int \frac{\mathrm{d} x}{a+b \sin x}=\frac{1}{\sqrt{b^{2}-a^{2}}} \ln \left|\frac{a \tan \frac{x}{2}+b-\sqrt{b^{2}-a^{2}}}{a \tan \frac{x}{2}+b+\sqrt{b^{2}-a^{2}}}\right|+C\left(a^{2}<b^{2}\right)$
105. $\int \frac{\mathrm{d} x}{a+b \cos x}=\frac{2}{a+b} \sqrt{\frac{a+b}{a-b}} \arctan \left(\sqrt{\frac{a-b}{a+b}} \tan \frac{x}{2}\right)+C\left(a^{2}>b^{2}\right)$
106. $\int \frac{\mathrm{d} x}{a+b \cos x}=\frac{1}{a+b} \sqrt{\frac{a+b}{b-a}} \ln \left|\frac{\tan \frac{x}{2}+\sqrt{\frac{a+b}{b-a}}}{\tan \frac{x}{2}-\sqrt{\frac{a+b}{b-a}}}\right|+C\left(a^{2}<b^{2}\right)$ 107. $\int \frac{\mathrm{d} x}{a^{2} \cos ^{2} x+b^{2} \sin ^{2} x}=\frac{1}{a b} \arctan \left(\frac{b}{a} \tan x\right)+C$
107. $\int \frac{\mathrm{d} x}{a^{2} \cos ^{2} x-b^{2} \sin ^{2} x}=\frac{1}{2 a b} \ln \left|\frac{b \tan x+a}{b \tan x-a}\right|+C$
108. $\int x \sin a x \mathrm{~d} x=\frac{1}{a^{2}} \sin a x-\frac{1}{a} x \cos a x+C$
109. $\int x^{2} \sin a x \mathrm{~d} x=-\frac{1}{a} x^{2} \cos a x+\frac{2}{a^{2}} x \sin a x+\frac{2}{a^{3}} \cos a x+C$
110. $\int x \cos a x \mathrm{~d} x=\frac{1}{a^{2}} \cos a x+\frac{1}{a} x \sin a x+C$
111. $\int x^{2} \cos a x \mathrm{~d} x=\frac{1}{a} x^{2} \sin a x+\frac{2}{a^{2}} x \cos a x-\frac{2}{a^{3}} \sin a x+C$

(十三) 含有反三角函数的积分 (其中 $a>0$ )

113. $\int \arcsin \frac{x}{a} \mathrm{~d} x=x \arcsin \frac{x}{a}+\sqrt{a^{2}-x^{2}}+C$
114. $\int x \arcsin \frac{x}{a} \mathrm{~d} x=\left(\frac{x^{2}}{2}-\frac{a^{2}}{4}\right) \arcsin \frac{x}{a}+\frac{x}{4} \sqrt{a^{2}-x^{2}}+C$
115. $\int x^{2} \arcsin \frac{x}{a} \mathrm{~d} x=\frac{x^{3}}{3} \arcsin \frac{x}{a}+\frac{1}{9}\left(x^{2}+2 a^{2}\right) \sqrt{a^{2}-x^{2}}+C$
116. $\int \arccos \frac{x}{a} \mathrm{~d} x=x \arccos \frac{x}{a}-\sqrt{a^{2}-x^{2}}+C$
117. $\int x \arccos \frac{x}{a} \mathrm{~d} x=\left(\frac{x^{2}}{2}-\frac{a^{2}}{4}\right) \arccos \frac{x}{a}-\frac{x}{4} \sqrt{a^{2}-x^{2}}+C$
118. $\int x^{2} \arccos \frac{x}{a} \mathrm{~d} x=\frac{x^{3}}{3} \arccos \frac{x}{a}-\frac{1}{9}\left(x^{2}+2 a^{2}\right) \sqrt{a^{2}-x^{2}}+C$
119. $\int \arctan \frac{x}{a} \mathrm{~d} x=x \arctan \frac{x}{a}-\frac{a}{2} \ln \left(a^{2}+x^{2}\right)+C$
120. $\int x \arctan \frac{x}{a} \mathrm{~d} x=\frac{1}{2}\left(a^{2}+x^{2}\right) \arctan \frac{x}{a}-\frac{a}{2} x+C$
121. $\int x^{2} \arctan \frac{x}{a} \mathrm{~d} x=\frac{x^{3}}{3} \arctan \frac{x}{a}-\frac{a}{6} x^{2}+\frac{a^{3}}{6} \ln \left(a^{2}+x^{2}\right)+C$

(十三) 含有指数函数的积分

122. $\int a^{x} \mathrm{~d} x=\frac{1}{\ln a} a^{x}+C$
123. $\int \mathrm{e}^{a x} \mathrm{~d} x=\frac{1}{a} \mathrm{e}^{a x}+C$ 124. $\int x \mathrm{e}^{u x} \mathrm{~d} x=\frac{1}{a^{2}}(a x-1) \mathrm{e}^{u x}+C$
124. $\int x^{n} \mathrm{e}^{a x} \mathrm{~d} x=\frac{1}{a} x^{n} \mathrm{e}^{a x}-\frac{n}{a} \int x^{n-1} \mathrm{e}^{a x} \mathrm{~d} x$
125. $\int x a^{\prime \prime} \mathrm{d} x=\frac{x}{\ln a} a^{x}-\frac{1}{(\ln a)^{2}} a^{x}+C$
126. $\int x^{n} a^{x} \mathrm{~d} x=\frac{1}{\ln a} x^{n} a^{x}-\frac{n}{\ln a} \int x^{n-1} a^{x} \mathrm{~d} x$
127. $\int \mathrm{e}^{a . r} \sin b x \mathrm{~d} x=\frac{1}{a^{2}+b^{2}} \mathrm{e}^{u, r}(a \sin b x-b \cos b x)+C$
128. $\int \mathrm{e}^{a x} \cos b x \mathrm{~d} x=\frac{1}{a^{2}+b^{2}} \mathrm{e}^{a x}(b \sin b x+a \cos b x)+C$
129. $\int \mathrm{e}^{a x} \sin ^{n} b x \mathrm{~d} x=\frac{1}{a^{2}+b^{2} n^{2}} \mathrm{e}^{a x} \sin ^{n-1} b x(a \sin b x-n b \cos b x)+$

$$
\frac{n(n-1) b^{2}}{a^{2}+b^{2} n^{2}} \int \mathrm{e}^{a x} \sin ^{n-2} b x \mathrm{~d} x
$$

131. $\int \mathrm{e}^{a x} \cos ^{n} b x \mathrm{~d} x=\frac{1}{a^{2}+b^{2} n^{2}} \mathrm{e}^{a x} \cos ^{n-1} b x(a \cos b x+n b \sin b x)+$

$$
\frac{n(n-1) b^{2}}{a^{2}+b^{2} n^{2}} \int \mathrm{e}^{a x} \cos ^{n-2} b x \mathrm{~d} x
$$

## (十四) 含有对数函数的积分

132. $\int \ln x \mathrm{~d} x=x \ln x-x+C$
133. $\int \frac{\mathrm{d} x}{x \ln x}=\ln |\ln x|+C$
134. $\int x^{n} \ln x \mathrm{~d} x=\frac{1}{n+1} x^{n+1}\left(\ln x-\frac{1}{n+1}\right)+C$
135. $\int(\ln x)^{n} \mathrm{~d} x=x(\ln x)^{n}-n \int(\ln x)^{n-1} \mathrm{~d} x$
136. $\int x^{\prime \prime \prime}(\ln x)^{\prime \prime} \mathrm{d} x=\frac{1}{m+1} x^{m+1}(\ln x)^{\prime \prime}-\frac{n}{m+1} \int x^{\prime \prime}(\ln x)^{n-1} \mathrm{~d} x$

## (十五) 含有双曲函数的积分

137. $\int \operatorname{sh} x \mathrm{~d} x=\operatorname{ch} x+C$
138. $\int \operatorname{ch} x \mathrm{~d} x=\operatorname{sh} x+C$
139. $\int$ th $x \mathrm{~d} x=\ln$ ch $x+C$ 140. $\int \operatorname{sh}^{2} x \mathrm{~d} x=-\frac{x}{2}+\frac{1}{4} \operatorname{sh} 2 x+C$
140. $\int \operatorname{ch}^{2} x \mathrm{~d} x=\frac{x}{2}+\frac{1}{4} \operatorname{sh} 2 x+C$

## (十六) 定积分

142. $\int_{-\pi}^{\pi} \cos n x \mathrm{~d} x=\int_{-\pi}^{\pi} \sin n x \mathrm{~d} x=0$
143. $\int_{-\pi}^{\pi} \cos m x \sin n x \mathrm{~d} x=0$
144. $\int_{-\pi}^{\pi} \cos m x \cos n x \mathrm{~d} x=\left\{\begin{array}{l}0, m \neq n \\ \pi, m=n\end{array}\right.$
145. $\int_{-\pi}^{\pi} \sin m x \sin n x \mathrm{~d} x= \begin{cases}0, & m \neq n \\ \pi, & m=n\end{cases}$
146. $\int_{0}^{\pi} \sin m x \sin n x \mathrm{~d} x=\int_{0}^{\pi} \cos m x \cos n x \mathrm{~d} x=\left\{\begin{array}{l}0, m \neq n \\ \pi / 2, m=n\end{array}\right.$
147. $I_{n}=\int_{0}^{\frac{\pi}{2}} \sin ^{n} x \mathrm{~d} x=\int_{0}^{\frac{\pi}{2}} \cos ^{n} x \mathrm{~d} x$

$$
\begin{aligned}
I_{n}= & \frac{n-1}{n} I_{n-2} \\
& =\left\{\begin{array}{l}
\frac{n-1}{n} \cdot \frac{n-3}{n-2} \cdots \cdots \cdot \frac{4}{5} \cdot \frac{2}{3} \text { ( } n \text { 为大于 } 1 \text { 的正奇数) }, I_{1}=1 \\
\frac{n-1}{n} \cdot \frac{n-3}{n-2} \cdots \cdot \frac{3}{4} \cdot \frac{1}{2} \cdot \frac{\pi}{2} \text { ( } n \text { 为正偶数) }, I_{0}=\frac{\pi}{2}
\end{array}\right.
\end{aligned}
$$

