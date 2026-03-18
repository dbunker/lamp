d2(o2).
d4(o2).
d0(o1).
d1(o4).
d2(o1).

d3(X) :- d4(X), d0(X), not d5(X).
d5(X) :- d1(X), d0(X), not d2(X).
d1(X) :- d4(X), d0(X), not d5(X).
d4(X) :- d1(X), d0(X), not d3(X).
d2(X) :- d1(X), not d5(X), d3(X).
d3(X) :- d0(X), d5(X), not d2(X).
d0(X) :- d2(X), d3(X), not d4(X).
d4(X) :- d2(X), d0(X), not d5(X).
d0(X) :- d5(X), d3(X), not d4(X).
d5(X) :- d2(X), d3(X), not d4(X).
d4(X) :- d2(X), d0(X), not d1(X).