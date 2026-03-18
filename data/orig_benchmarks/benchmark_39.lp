d9(o5).
d1(o4).
d9(o4).
d2(o0).
d7(o2).
d5(o3).

d3(X) :- d9(X), d8(X).
d5(X) :- d9(X), d8(X).
d6(X) :- d5(X), d7(X).
d1(X) :- d9(X), d5(X).
d0(X) :- d2(X), d5(X).
d8(X) :- d9(X), d6(X).
d7(X) :- d3(X), d1(X).
d1(X) :- d2(X), d7(X).
d7(X) :- d9(X), d1(X).
d2(X) :- d9(X), d5(X).
d9(X) :- d6(X), d5(X).