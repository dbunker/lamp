d4(o1).
d0(o2).
d2(o4).
d4(o5).
d3(o4).
d4(o3).
d1(o2).
d0(o3).
d0(o4).

d2(X) :- d1(X), d0(X), d3(X).
d1(X) :- d2(X), d0(X), d5(X).
d5(X) :- d4(X), d0(X), d2(X).
d2(X) :- d4(X), d5(X), d3(X).
d4(X) :- d0(X), d5(X), d3(X).
d2(X) :- d4(X), d5(X), d3(X).
d1(X) :- d2(X), d0(X), d4(X).
d0(X) :- d2(X), d1(X), d5(X).
d2(X) :- d4(X), d1(X), d3(X).
d1(X) :- d2(X), d0(X), d3(X).
d4(X) :- d2(X), d0(X), d3(X).