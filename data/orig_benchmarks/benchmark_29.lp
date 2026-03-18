d6(o1).
d4(o0).
d7(o1).
d3(o3).
d9(o1).
d5(o1).
d2(o2).
d8(o1).
d4(o3).

d6(X) :- d9(X), d4(X).
d1(X) :- d0(X), d5(X).
d7(X) :- d9(X), d0(X).
d2(X) :- d6(X), d7(X).
d4(X) :- d9(X), d8(X).
d2(X) :- d6(X), d5(X).
d0(X) :- d9(X), d3(X).
d3(X) :- d0(X), d8(X).
d7(X) :- d5(X), d4(X).
d0(X) :- d3(X), d6(X).
d0(X) :- d9(X), d2(X).