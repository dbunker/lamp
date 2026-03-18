d5(o1).
d1(o3).
d2(o5).
d0(o5).
d5(o5).
d4(o3).

d4(X) :- d2(X), d0(X), d5(X).
d2(X) :- d4(X), d0(X), d5(X).
d1(X) :- d4(X), d0(X), d3(X).
d3(X) :- d4(X), d0(X), d2(X).
d4(X) :- d2(X), d0(X), d5(X).
d4(X) :- d2(X), d5(X), d3(X).
d4(X) :- d2(X), d0(X), d5(X).
d4(X) :- d1(X), d5(X), d3(X).
d3(X) :- d2(X), d0(X), d5(X).
d0(X) :- d4(X), d1(X), d5(X).
d0(X) :- d2(X), d1(X), d5(X).