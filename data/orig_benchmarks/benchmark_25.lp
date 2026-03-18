d9(o5).
d4(o5).
d8(o3).
d2(o0).
d5(o1).
d8(o5).
d4(o1).
d2(o1).
d3(o5).

d4(X) :- d2(X), d8(X), d1(X).
d1(X) :- d0(X), d3(X), d5(X).
d2(X) :- d9(X), d3(X), d4(X).
d9(X) :- d2(X), d3(X), d4(X).
d2(X) :- d6(X), d5(X), d4(X).
d8(X) :- d2(X), d5(X), d4(X).
d9(X) :- d2(X), d3(X), d1(X).
d1(X) :- d0(X), d3(X), d8(X).
d7(X) :- d2(X), d3(X), d5(X).
d6(X) :- d5(X), d9(X), d7(X).
d0(X) :- d5(X), d1(X), d4(X).