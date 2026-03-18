d2(o3).
d5(o4).
d3(o3).
d2(o1).
d1(o3).
d3(o2).

d4(X) :- d2(X), d3(X), d1(X).
d4(X) :- d2(X), d3(X), d1(X).
d2(X) :- d0(X), d3(X), d5(X).
d1(X) :- d2(X), d5(X), d4(X).
d4(X) :- d0(X), d5(X), d1(X).
d1(X) :- d0(X), d2(X), d5(X).
d1(X) :- d0(X), d2(X), d4(X).
d4(X) :- d0(X), d2(X), d3(X).
d0(X) :- d2(X), d3(X), d5(X).
d5(X) :- d3(X), d1(X), d4(X).
d3(X) :- d2(X), d1(X), d4(X).