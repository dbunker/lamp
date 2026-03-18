d7(o0).
d1(o4).
d2(o2).
d3(o3).
d4(o4).
d4(o1).
d9(o2).
d3(o4).
d8(o4).

d3(X) :- d9(X), d2(X), d1(X).
d5(X) :- d0(X), d3(X), d7(X).
d8(X) :- d5(X), d1(X), d7(X).
d5(X) :- d9(X), d0(X), d7(X).
d0(X) :- d9(X), d3(X), d8(X).
d7(X) :- d9(X), d3(X), d8(X).
d0(X) :- d1(X), d7(X), d8(X).
d1(X) :- d9(X), d2(X), d8(X).
d7(X) :- d8(X), d1(X), d4(X).
d1(X) :- d0(X), d2(X), d5(X).
d3(X) :- d5(X), d1(X), d4(X).