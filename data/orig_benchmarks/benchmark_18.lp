d2(o5).
d4(o1).
d4(o2).
d0(o2).
d7(o3).
d7(o5).

d5(X) :- d9(X), d4(X).
d7(X) :- d0(X), d4(X).
d9(X) :- d0(X), d7(X).
d1(X) :- d0(X), d3(X).
d6(X) :- d3(X), d4(X).
d2(X) :- d3(X), d5(X).
d8(X) :- d9(X), d4(X).
d8(X) :- d2(X), d5(X).
d3(X) :- d9(X), d6(X).
d7(X) :- d6(X), d8(X).
d9(X) :- d6(X), d8(X).