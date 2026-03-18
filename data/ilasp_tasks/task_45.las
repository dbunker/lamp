#constant(obj, o1).
#constant(obj, o4).
#constant(obj, o6).
#constant(obj, o8).
#constant(obj, o9).

#modeh(d1(var(obj))).
#modeh(d2(var(obj))).
#modeh(d3(var(obj))).

#modeb(1, d0(var(obj))).
#modeb(1, d0(var(obj)), (negative)).
#modeb(1, d1(var(obj))).
#modeb(1, d1(var(obj)), (negative)).
#modeb(1, d2(var(obj))).
#modeb(1, d2(var(obj)), (negative)).
#modeb(1, d3(var(obj))).
#modeb(1, d3(var(obj)), (negative)).
#modeb(1, d4(var(obj))).
#modeb(1, d4(var(obj)), (negative)).
#modeb(1, d5(var(obj))).
#modeb(1, d5(var(obj)), (negative)).

#pos(eg1, {
  d1(o8), d2(o8), d3(o8)
}, {
  d1(o1), d1(o4), d1(o6), d2(o1), d2(o4), d2(o9), d3(o1), d3(o4), d3(o6)
}, {
  d0(o1).
  d0(o4).
  d0(o8).
  d1(o9).
  d2(o6).
  d3(o9).
  d4(o4).
  d4(o8).
  d5(o8).
}).