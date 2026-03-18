d4(o1).
d4(o2).
d1(o5).
d0(o2).
d4(o5).
d4(o3).

d3(X) :- d2(X), d0(X).
d5(X) :- d4(X), d1(X).
d2(X) :- d1(X), d3(X).
d1(X) :- d0(X), d5(X).
d5(X) :- d2(X), d4(X).
d1(X) :- d0(X), d3(X).
d3(X) :- d4(X), d0(X).
d4(X) :- d2(X), d5(X).
d0(X) :- d2(X), d5(X).
d4(X) :- d2(X), d5(X).
d4(X) :- d1(X), d0(X).